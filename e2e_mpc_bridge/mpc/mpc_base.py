"""
mpc_base.py
===========
MPC コントローラの抽象基底 (レポート第5章の MpcControllerBase 実装)。

責務:
- update_plan(低レート, 非同期): E2E 出力を内部参照として受領
- step(高レート, 同期): 保持中の参照に追従する1ステップ最適化
- emergency_stop: E2E 失効・ソルバ失敗時の安全減速

スレッド安全性:
  update_plan と step は別スレッド(推論10-20Hz vs 制御50-100Hz)から呼ばれる。
  ダブルバッファ + ロックで、step が参照する PlanningTarget の一貫性を保つ。
  ロック保持は「参照ポインタの差し替え」だけに限定し、最適化計算はロック外で行う。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import threading
import time

from contract import (
    PlanningTarget, VehicleState, ControlCommand, MpcStatus,
    FallbackState, DrivingMode,
)


def now_ns() -> int:
    return time.monotonic_ns()


class MpcControllerBase(ABC):
    """Acados MPC ラッパの抽象基底。

    Parameters
    ----------
    plan_timeout_s : float
        step 時刻と target 時刻の差がこれを超えたら STALE_PLAN。
    emergency_decel : float
        フォールバック時の減速度 [m/s^2] (正値)。
    """

    def __init__(self, plan_timeout_s: float = 0.5,
                 emergency_decel: float = 4.0):
        self.plan_timeout_s = plan_timeout_s
        self.emergency_decel = emergency_decel
        self._lock = threading.Lock()
        self._target: PlanningTarget | None = None
        self._last_plan_id: int = -1

    # ---- 低レート・非同期 ----
    def update_plan(self, target: PlanningTarget) -> None:
        """最新の E2E 出力を受領 (ダブルバッファのポインタ差し替えのみ)。"""
        with self._lock:
            self._target = target

    # ---- 高レート・同期 ----
    def step(self, state: VehicleState) -> tuple[ControlCommand, MpcStatus]:
        """1 ステップ最適化。失効/実行不能時はフォールバックを返す。"""
        with self._lock:
            target = self._target   # 参照取得はロック内

        # 1) 失効検出
        if target is None:
            return self._fallback(state, FallbackState.STALE_PLAN, plan_id=-1)
        age_s = (state.stamp_ns - target.stamp_ns) * 1e-9
        if age_s > self.plan_timeout_s:
            return self._fallback(state, FallbackState.STALE_PLAN,
                                  plan_id=target.plan_id)

        # 2) EMERGENCY モードは即フォールバック
        if target.mode == DrivingMode.EMERGENCY:
            return self._fallback(state, FallbackState.NONE,
                                  plan_id=target.plan_id, emergency=True)

        # 3) 最適化 (ロック外で実行)
        t0 = time.perf_counter()
        try:
            cmd, status = self._solve(state, target)
        except Exception:                       # ソルバ内部例外
            return self._fallback(state, FallbackState.SOLVER_FAIL,
                                  plan_id=target.plan_id)
        status.solve_time_ms = (time.perf_counter() - t0) * 1e3
        self._last_plan_id = target.plan_id
        return cmd, status

    # ---- フォールバック生成 ----
    def _fallback(self, state: VehicleState, reason: FallbackState,
                  plan_id: int, emergency: bool = False
                  ) -> tuple[ControlCommand, MpcStatus]:
        cmd = self.emergency_stop(state)
        cmd.plan_id = plan_id
        st = MpcStatus(
            stamp_ns=now_ns(), solve_ok=False, solve_time_ms=0.0,
            fallback=(FallbackState.NONE if emergency else reason),
        )
        return cmd, st

    def emergency_stop(self, state: VehicleState) -> ControlCommand:
        """安全減速プロファイル: 現在舵角維持 + 一定減速で停止へ。"""
        decel = -abs(self.emergency_decel)
        if state.v <= 0.05:      # ほぼ停止していれば減速指令を 0 に
            decel = 0.0
        return ControlCommand(
            stamp_ns=now_ns(), plan_id=self._last_plan_id,
            steer_angle=state.steer_angle, accel=decel, jerk=0.0)

    # ---- サブクラスが実装 ----
    @abstractmethod
    def _solve(self, state: VehicleState, target: PlanningTarget
               ) -> tuple[ControlCommand, MpcStatus]:
        """参照 target に state を追従させる1ステップ最適化。"""
        ...

    def close(self) -> None:
        return None
