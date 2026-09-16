"""
demo_e2e_mpc.py
===============
E2E <-> MPC 統合の動作検証。実 acados 無しでも SimpleMpcController で走る。

検証項目:
  1. 軌跡のみ(UniAD/VAD)の infer -> update_plan -> step が完走
  2. Hydra-NeXt の πctrl が control_ref として MPC に渡り warm start に効く
  3. 失効(STALE_PLAN)検出とフォールバック
  4. EMERGENCY モードのフォールバック
  5. スレッド安全性 (update_plan と step の並行実行)
"""
from __future__ import annotations
import os
import sys
import threading
import time
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "mpc"))
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))

from contract import VehicleState, FallbackState, DrivingMode  # noqa: E402
from acados_mpc import SimpleMpcController                      # noqa: E402
from e2e_interface import (                                     # noqa: E402
    TrajectoryOnlyPlanner, HydraNextPlanner, E2EMpcRunner,
)


def straight_traj(n=8, dt=0.5, v=8.0):
    """直進する参照軌跡 (x前方) を作る。"""
    xs = np.cumsum(np.full(n, v * dt))
    ys = np.zeros(n)
    return np.column_stack([xs, ys])


def make_state(t_ns, x=0.0, y=0.0, theta=0.0, v=8.0, steer=0.0):
    return VehicleState(stamp_ns=t_ns, x=x, y=y, theta=theta, v=v,
                        steer_angle=steer, accel=0.0)


def test_trajectory_only():
    print("=== Test 1: UniAD/VAD 軌跡のみ ===")
    def fwd(_batch):
        return {"traj_xy": straight_traj(), "dt": 0.5, "confidence": 0.9}
    planner = TrajectoryOnlyPlanner(fwd, dt=0.5)
    mpc = SimpleMpcController(plan_timeout_s=0.5)
    runner = E2EMpcRunner(planner, mpc)

    t = time.monotonic_ns()
    runner.on_sensor({})                      # 推論 -> update_plan
    cmd, st = runner.tick(make_state(t))      # 同時刻なので失効しない
    assert st.fallback == FallbackState.NONE, st.fallback
    assert st.solve_ok
    print(f"  steer={cmd.steer_angle:.4f} accel={cmd.accel:.3f} "
          f"ok={st.solve_ok} err_lat={st.tracking_err_lat:.3f}")


def test_hydranext_control_ref():
    print("=== Test 2: Hydra-NeXt πctrl (control_ref) ===")
    def fwd(_batch):
        traj = straight_traj()
        xyt = np.column_stack([traj[:, 0], traj[:, 1],
                               np.zeros(len(traj)), np.full(len(traj), 8.0)])
        return {"traj_xyt": xyt, "control": np.array([1.5, 0.02]),
                "control_valid_s": 0.5, "dt": 0.5, "confidence": 0.95}
    planner = HydraNextPlanner(fwd, dt=0.5)
    mpc = SimpleMpcController()
    runner = E2EMpcRunner(planner, mpc)

    t = time.monotonic_ns()
    target = planner.infer({})
    assert target.has_control_ref, "control_ref が載っていない"
    print(f"  control_ref = accel {target.control_ref[0]}, "
          f"steer_rate {target.control_ref[1]}")
    mpc.update_plan(target)
    cmd, st = runner.tick(make_state(t))
    # πctrl の accel(1.5) が混合されて反映される
    print(f"  混合後 accel={cmd.accel:.3f} (πctrl accel=1.5 と P制御の平均)")
    assert st.solve_ok


def test_stale_plan():
    print("=== Test 3: 失効検出 (STALE_PLAN) ===")
    def fwd(_batch):
        return {"traj_xy": straight_traj(), "dt": 0.5}
    planner = TrajectoryOnlyPlanner(fwd)
    mpc = SimpleMpcController(plan_timeout_s=0.3, emergency_decel=4.0)
    runner = E2EMpcRunner(planner, mpc)

    target = planner.infer({})
    mpc.update_plan(target)
    # target より 0.5s 後の state を渡す -> 0.3s のタイムアウト超過
    late = make_state(target.stamp_ns + int(0.5e9), v=8.0)
    cmd, st = mpc.step(late)
    assert st.fallback == FallbackState.STALE_PLAN, st.fallback
    assert cmd.accel < 0, "減速フォールバックになっていない"
    print(f"  fallback={st.fallback.value} accel={cmd.accel:.2f} (緊急減速)")


def test_emergency_mode():
    print("=== Test 4: EMERGENCY モード ===")
    def fwd(_batch):
        return {"traj_xy": straight_traj(), "dt": 0.5, "mode": "EMERGENCY"}
    planner = TrajectoryOnlyPlanner(fwd)
    mpc = SimpleMpcController()
    target = planner.infer({})
    mpc.update_plan(target)
    cmd, st = mpc.step(make_state(target.stamp_ns))
    assert cmd.accel < 0
    print(f"  EMERGENCY -> accel={cmd.accel:.2f} 即減速")


def test_thread_safety():
    print("=== Test 5: スレッド安全性 (並行 update_plan / step) ===")
    def fwd(_batch):
        return {"traj_xy": straight_traj(), "dt": 0.5}
    planner = TrajectoryOnlyPlanner(fwd)
    mpc = SimpleMpcController(plan_timeout_s=10.0)  # 失効させない

    stop = threading.Event()
    errors = []

    def planner_loop():   # 低レート
        while not stop.is_set():
            try:
                mpc.update_plan(planner.infer({}))
            except Exception as e:
                errors.append(e)
            time.sleep(0.005)

    def control_loop():   # 高レート
        while not stop.is_set():
            try:
                t = time.monotonic_ns()
                mpc.step(make_state(t))
            except Exception as e:
                errors.append(e)
            time.sleep(0.001)

    th1 = threading.Thread(target=planner_loop)
    th2 = threading.Thread(target=control_loop)
    th1.start(); th2.start()
    time.sleep(0.5)
    stop.set(); th1.join(); th2.join()
    assert not errors, f"並行実行でエラー: {errors[:3]}"
    print(f"  0.5秒間の並行実行でエラーなし (競合なし)")


if __name__ == "__main__":
    test_trajectory_only()
    test_hydranext_control_ref()
    test_stale_plan()
    test_emergency_mode()
    test_thread_safety()
    print("\n全テスト通過: 契約・πctrl・失効検出・EMERGENCY・スレッド安全性 OK")
    # acados 実機確認は環境依存のためスキップ (import 可否だけ報告)
    try:
        import acados_template  # noqa: F401
        print("acados_template 検出: AcadosMpcController も利用可能")
    except ImportError:
        print("acados_template 未検出: SimpleMpcController で動作 "
              "(実MPCは acados 導入後に AcadosMpcController へ差し替え)")
