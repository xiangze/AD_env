"""
contract.py
===========
E2E Planner <-> MPC の境界を固定するデータ契約 (レポート第5章の Python 実装)。

設計原則:
- torch/acados に非依存の純粋 dataclass + numpy。言語境界(Protobuf)や
  プロセス境界(RPC)に載せられるよう、副作用を持たない値オブジェクトにする。
- 座標系は ego 後輪軸中心 (x:前方+, y:左方+, theta:反時計+)、SI 単位。
- すべてのメッセージが stamp(ns) を持ち、時刻整合(失効検出)を可能にする。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DrivingMode(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    EMERGENCY = "EMERGENCY"


class FallbackState(str, Enum):
    NONE = "NONE"
    STALE_PLAN = "STALE_PLAN"     # E2E 出力が古い
    INFEASIBLE = "INFEASIBLE"     # 最適化が実行不能
    SOLVER_FAIL = "SOLVER_FAIL"   # ソルバ非収束


# ---------------------------------------------------------------------------
# E2E -> MPC
# ---------------------------------------------------------------------------
@dataclass
class PlanningTarget:
    """E2E Planner の出力。参照軌跡 + 任意の制御参照(πctrl)。

    trajectory: (N, 6) 各行 [t, x, y, theta, v, kappa], 先頭 t=0 が現在。
    control_ref: (2,) [accel, steer_rate] または None (πctrl 未使用時)。
    """
    stamp_ns: int
    plan_id: int
    dt_s: float
    trajectory: np.ndarray
    confidence: float = 1.0
    mode: DrivingMode = DrivingMode.NORMAL
    target_speed: float = 0.0
    control_ref: np.ndarray | None = None
    control_ref_valid_until_s: float = 0.0

    def __post_init__(self):
        self.trajectory = np.asarray(self.trajectory, dtype=np.float64)
        if self.trajectory.ndim != 2 or self.trajectory.shape[1] != 6:
            raise ValueError(
                f"trajectory は (N,6) が必要: got {self.trajectory.shape}")
        if self.control_ref is not None:
            self.control_ref = np.asarray(self.control_ref, dtype=np.float64)
            if self.control_ref.shape != (2,):
                raise ValueError("control_ref は (2,) [accel, steer_rate]")

    @property
    def has_control_ref(self) -> bool:
        return self.control_ref is not None

    @property
    def horizon_s(self) -> float:
        return float(self.trajectory[-1, 0]) if len(self.trajectory) else 0.0


@dataclass
class VehicleState:
    """MPC への現在自己状態 (ego 後輪軸中心)。"""
    stamp_ns: int
    x: float
    y: float
    theta: float
    v: float
    steer_angle: float
    accel: float = 0.0

    def as_array(self) -> np.ndarray:
        # kinematic bicycle の状態順 [x, y, theta, v] に合わせる
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float64)


# ---------------------------------------------------------------------------
# MPC -> Vehicle / 監視系
# ---------------------------------------------------------------------------
@dataclass
class ControlCommand:
    """MPC が出すアクチュエータ指令。"""
    stamp_ns: int
    plan_id: int
    steer_angle: float   # 前輪舵角 [rad]
    accel: float         # 縦加速度 [m/s^2]
    jerk: float = 0.0


@dataclass
class MpcStatus:
    """MPC の状態フィードバック (監視・E2E への逆通知)。"""
    stamp_ns: int
    solve_ok: bool
    solve_time_ms: float
    tracking_err_lat: float = 0.0
    tracking_err_lon: float = 0.0
    sqp_iter: int = 0
    fallback: FallbackState = FallbackState.NONE
    cost: float = 0.0
