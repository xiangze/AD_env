#!/usr/bin/env python3
"""
run_cosim.py — E2E プランナ軌道 → acados 追従 NMPC の協調実行(replay モード)

前提:
  E2E(UniAD/VAD)の評価/推論が、各フレームの将来 ego 軌道をファイルに書き出している。
  本スクリプトはその軌道列を読み、uniad_vad_acados_mpc の追従 NMPC に通して
  制御量 [a, delta] の時系列を出力する(閉ループ風の replay)。

入力フォーマット(例):
  --plan <path>  : npz もしくは json。
    npz: key 'waypoints' = (T_frames, M, 2) ego 座標系 waypoint 列(各フレーム M 点)
    json: [{"waypoints": [[x,y],...], "v": <float>}, ...]

出力:
  --out <path.csv> : frame, a[m/s^2], delta[rad], solver_status

使い方:
  python run_cosim.py --plan /data/plan.npz --out /out/control_trace.csv \
      --N 20 --Tf 3.0 --wheelbase 2.9
"""
import argparse, csv, json, sys
import numpy as np

from uniad_vad_acados_mpc import build_solver, plan_to_reference, step


def load_plan(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        wp = d["waypoints"]              # (T, M, 2)
        v0 = d["v"] if "v" in d else None
        return [(wp[i], None if v0 is None else float(v0[i])) for i in range(len(wp))]
    with open(path) as f:
        frames = json.load(f)
    return [(np.asarray(fr["waypoints"], float), fr.get("v")) for fr in frames]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--Tf", type=float, default=3.0)
    p.add_argument("--wheelbase", type=float, default=2.9)
    args = p.parse_args()

    solver = build_solver(args.N, args.Tf)   # OCP 生成 + C コンパイル(初回のみ時間)
    frames = load_plan(args.plan)

    # 単純な前進積分で疑似的な ego 状態を更新しながら各フレームを解く
    x = np.array([0.0, 0.0, 0.0, frames[0][1] or 8.0])   # [X,Y,psi,v]
    rows = []
    for i, (wp, v_meas) in enumerate(frames):
        v_cur = v_meas if v_meas is not None else x[3]
        ref = plan_to_reference(wp, args.N, args.Tf, v_current=v_cur)
        u0, status = step(solver, x, ref, args.N)
        a, delta = float(u0[0]), float(u0[1])
        rows.append([i, a, delta, int(status)])
        # 1 ステップ(Tf/N 秒)だけ運動学で状態を進める(replay の簡易前進)
        dt = args.Tf / args.N
        X, Y, psi, v = x
        x = np.array([
            X + v * np.cos(psi) * dt,
            Y + v * np.sin(psi) * dt,
            psi + v / args.wheelbase * np.tan(delta) * dt,
            v + a * dt,
        ])
        if status != 0:
            print(f"[warn] frame {i}: acados status {status}", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "a_mps2", "delta_rad", "solver_status"])
        w.writerows(rows)
    ok = sum(1 for r in rows if r[3] == 0)
    print(f"done: {len(rows)} frames, {ok} solved OK -> {args.out}")


if __name__ == "__main__":
    main()
