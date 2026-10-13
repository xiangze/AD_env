#!/usr/bin/env python3
"""
run_cosim.py — E2E プランナ軌道 → 追従制御器(acados MPC または LQR)の協調実行(replay)

E2E(UniAD/VAD)が各フレームで書き出した将来 ego 軌道を読み、追従制御器に通して
制御量 [a, delta] の時系列を出力する。制御器は --controller で切替:
    acados : 拘束付き追従 NMPC(C コード生成が必要。高機能)
    lqr    : 線形化 + 離散 Riccati の時変 LQR(numpy/scipy のみ。即実行)

入力(例):
    --plan <path.npz>  key 'waypoints' = (T_frames, M, 2) ego 座標系, 任意で 'v'
    --plan <path.json> [{"waypoints": [[x,y],...], "v": <float>}, ...]

出力:
    --out <path.csv>   frame, a[m/s^2], delta[rad], solver_status, controller

使い方:
    python run_cosim.py --controller lqr   --plan /data/plans/uniad_plan.npz --out /out/trace.csv
    python run_cosim.py --controller acados --plan /data/plans/uniad_plan.npz --out /out/trace.csv
"""
import argparse, csv, json, sys
import numpy as np


def load_backend(controller):
    """制御器に応じて必要なモジュールだけ遅延 import する。
    lqr のときは acados を一切読み込まないので、acados ビルド無しで動く。"""
    if controller == "acados":
        from uniad_vad_acados_mpc import build_solver, plan_to_reference, step
        return build_solver, plan_to_reference, step
    elif controller == "lqr":
        from lqr_tracker import build_tracker, plan_to_reference, step
        return build_tracker, plan_to_reference, step
    raise ValueError(f"unknown controller: {controller}")


def load_plan(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        wp = d["waypoints"]
        v0 = d["v"] if "v" in d else None
        return [(wp[i], None if v0 is None else float(v0[i])) for i in range(len(wp))]
    with open(path) as f:
        frames = json.load(f)
    return [(np.asarray(fr["waypoints"], float), fr.get("v")) for fr in frames]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["acados", "lqr"], default="lqr")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--Tf", type=float, default=3.0)
    p.add_argument("--wheelbase", type=float, default=2.9)
    args = p.parse_args()

    build, plan_to_reference, step = load_backend(args.controller)
    # acados の build_solver(N,Tf) と LQR の build_tracker(N,Tf,L) で引数が違うため分岐
    if args.controller == "acados":
        ctrl = build(args.N, args.Tf)
    else:
        ctrl = build(args.N, args.Tf, args.wheelbase)

    frames = load_plan(args.plan)
    x = np.array([0.0, 0.0, 0.0, frames[0][1] or 8.0])   # [X,Y,psi,v]
    rows = []
    for i, (wp, v_meas) in enumerate(frames):
        v_cur = v_meas if v_meas is not None else x[3]
        ref = plan_to_reference(wp, args.N, args.Tf, v_current=v_cur)
        u0, status = step(ctrl, x, ref, args.N)
        a, delta = float(u0[0]), float(u0[1])
        rows.append([i, a, delta, int(status), args.controller])
        dt = args.Tf / args.N
        X, Y, psi, v = x
        x = np.array([
            X + v * np.cos(psi) * dt,
            Y + v * np.sin(psi) * dt,
            psi + v / args.wheelbase * np.tan(delta) * dt,
            v + a * dt,
        ])
        if status != 0:
            print(f"[warn] frame {i}: {args.controller} status {status}", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "a_mps2", "delta_rad", "solver_status", "controller"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[3] == 0)
    print(f"done [{args.controller}]: {len(rows)} frames, {ok} solved OK -> {args.out}")


if __name__ == "__main__":
    main()
