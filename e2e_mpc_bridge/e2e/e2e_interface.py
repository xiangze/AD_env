"""
e2e_interface.py
================
E2E Planner 側のインターフェース (レポート第5章の E2EPlannerBase 実装)。

責務:
- E2E モデル(UniAD/VAD/Hydra-NeXt)の生出力を PlanningTarget(契約)へ変換する
- モデル固有部分は adapter に閉じ込め、MPC 側は契約しか知らない
- Hydra-NeXt の πctrl 出力を control_ref にマッピングする

境界の思想: E2E 側を UniAD -> VAD -> Hydra-NeXt と差し替えても、
出力が PlanningTarget に正規化されていれば MPC 側は一切変更不要。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np

from contract import PlanningTarget, DrivingMode


def now_ns() -> int:
    return time.monotonic_ns()


class E2EPlannerBase(ABC):
    """BEV バックボーン + Planner ヘッドのラッパの抽象基底。

    infer() がセンサ入力から PlanningTarget を返す。
    """

    def __init__(self):
        self._plan_counter = 0

    @abstractmethod
    def _forward(self, sensor_batch: dict) -> dict:
        """モデル固有の推論。生出力(dict)を返す。サブクラスが実装。"""
        ...

    @abstractmethod
    def _to_planning_target(self, raw: dict) -> PlanningTarget:
        """生出力 -> PlanningTarget。サブクラスが実装。"""
        ...

    def infer(self, sensor_batch: dict) -> PlanningTarget:
        raw = self._forward(sensor_batch)
        target = self._to_planning_target(raw)
        return target

    def _next_plan_id(self) -> int:
        self._plan_counter += 1
        return self._plan_counter


# ---------------------------------------------------------------------------
# UniAD / VAD: 軌跡のみ出力 (control_ref なし)
# ---------------------------------------------------------------------------
class TrajectoryOnlyPlanner(E2EPlannerBase):
    """UniAD/VAD のような軌跡のみ出力のモデル用アダプタ。

    forward_fn: sensor_batch -> {"traj_xy": (N,2), "dt": float,
                                 "confidence": float}
    をコールバックで受け取り、契約に正規化する。
    軌跡から theta, v, kappa を差分で補完する。
    """

    def __init__(self, forward_fn, dt: float = 0.5):
        super().__init__()
        self._forward_fn = forward_fn
        self._dt = dt

    def _forward(self, sensor_batch: dict) -> dict:
        return self._forward_fn(sensor_batch)

    def _to_planning_target(self, raw: dict) -> PlanningTarget:
        xy = np.asarray(raw["traj_xy"], dtype=np.float64)   # (N,2)
        dt = raw.get("dt", self._dt)
        n = len(xy)
        t = np.arange(n) * dt
        # theta: 進行方向を差分から推定
        dxy = np.diff(xy, axis=0, prepend=xy[:1])
        theta = np.arctan2(dxy[:, 1], dxy[:, 0])
        theta[0] = theta[1] if n > 1 else 0.0
        # v: 変位/dt
        v = np.linalg.norm(dxy, axis=1) / dt
        v[0] = v[1] if n > 1 else 0.0
        # kappa: 向き変化/距離
        dtheta = np.diff(theta, prepend=theta[:1])
        ds = np.maximum(np.linalg.norm(dxy, axis=1), 1e-3)
        kappa = dtheta / ds
        traj = np.column_stack([t, xy[:, 0], xy[:, 1], theta, v, kappa])
        return PlanningTarget(
            stamp_ns=now_ns(), plan_id=self._next_plan_id(), dt_s=dt,
            trajectory=traj, confidence=raw.get("confidence", 1.0),
            mode=DrivingMode(raw.get("mode", "NORMAL")),
            target_speed=float(v.max()) if n else 0.0)


# ---------------------------------------------------------------------------
# Hydra-NeXt: πtraj + πctrl 出力 (control_ref あり)
# ---------------------------------------------------------------------------
class HydraNextPlanner(E2EPlannerBase):
    """Hydra-NeXt 用アダプタ。πtraj を軌跡に、πctrl を control_ref に載せる。

    forward_fn: sensor_batch -> {
        "traj_xyt": (N,4) [x,y,theta,v],  # πtraj
        "control": (2,) [accel, steer_rate],  # πctrl
        "control_valid_s": float,
        "dt": float, "confidence": float, "mode": str,
    }
    """

    def __init__(self, forward_fn, dt: float = 0.5):
        super().__init__()
        self._forward_fn = forward_fn
        self._dt = dt

    def _forward(self, sensor_batch: dict) -> dict:
        return self._forward_fn(sensor_batch)

    def _to_planning_target(self, raw: dict) -> PlanningTarget:
        arr = np.asarray(raw["traj_xyt"], dtype=np.float64)  # (N,4)
        dt = raw.get("dt", self._dt)
        n = len(arr)
        t = (np.arange(n) * dt).reshape(-1, 1)
        # kappa を dtheta/ds で補完
        theta = arr[:, 2]
        xy = arr[:, :2]
        dxy = np.diff(xy, axis=0, prepend=xy[:1])
        dtheta = np.diff(theta, prepend=theta[:1])
        ds = np.maximum(np.linalg.norm(dxy, axis=1), 1e-3)
        kappa = (dtheta / ds).reshape(-1, 1)
        traj = np.hstack([t, arr, kappa])   # [t,x,y,theta,v,kappa]

        control = np.asarray(raw["control"], dtype=np.float64)
        return PlanningTarget(
            stamp_ns=now_ns(), plan_id=self._next_plan_id(), dt_s=dt,
            trajectory=traj, confidence=raw.get("confidence", 1.0),
            mode=DrivingMode(raw.get("mode", "NORMAL")),
            target_speed=float(arr[:, 3].max()) if n else 0.0,
            control_ref=control,
            control_ref_valid_until_s=raw.get("control_valid_s", dt))


# ---------------------------------------------------------------------------
# E2E -> MPC を繋ぐランループ (別レートのスレッド駆動を模した同期版)
# ---------------------------------------------------------------------------
class E2EMpcRunner:
    """E2E(低レート) と MPC(高レート) を繋ぐ配線。

    実運用では 2 スレッド:
      - planner_thread: infer() -> mpc.update_plan()  (10-20Hz)
      - control_thread: mpc.step(state) -> actuator    (50-100Hz)
    ここでは決定論的にテストするため tick() で 1 制御周期を回す同期版を提供。
    """

    def __init__(self, planner: E2EPlannerBase, mpc):
        self.planner = planner
        self.mpc = mpc

    def on_sensor(self, sensor_batch: dict) -> None:
        """低レート: 新しい推論結果を MPC に渡す。"""
        target = self.planner.infer(sensor_batch)
        self.mpc.update_plan(target)

    def tick(self, state):
        """高レート: 1 制御周期。"""
        return self.mpc.step(state)
