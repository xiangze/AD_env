"""
acados_mpc.py
=============
MpcControllerBase の Acados 実装。

車両モデル: kinematic bicycle
  状態 x = [X, Y, psi, v]         (後輪軸中心)
  入力 u = [a, delta]             (縦加速度, 前輪舵角)
  dX   = v cos(psi)
  dY   = v sin(psi)
  dpsi = v / L * tan(delta)
  dv   = a

目的: E2E 参照軌跡 (x,y,psi,v) への追従 + 入力最小化。
制約: 舵角・加速度の上下限、舵角レート(スルー rate)。
warm start: PlanningTarget.control_ref(πctrl) があれば初期入力に使う
            → Hydra-NeXt の πctrl を「MPC が運動学的実行可能性を保証する初期解」
              として活かす (レポート第3章 πdp 相当の役割を MPC が担保)。

実 acados が無い環境では ImportError を投げるので、フォールバックの
SimpleMpcController(同ファイル下部) を使うこと。
"""
from __future__ import annotations
import numpy as np

from contract import (
    PlanningTarget, VehicleState, ControlCommand, MpcStatus, FallbackState,
)
from mpc_base import MpcControllerBase, now_ns


class AcadosMpcController(MpcControllerBase):
    def __init__(self, wheelbase: float = 2.7, horizon_n: int = 20,
                 dt: float = 0.1, steer_max: float = 0.6,
                 accel_bounds: tuple[float, float] = (-5.0, 3.0),
                 steer_rate_max: float = 0.4, **kw):
        super().__init__(**kw)
        self.L = wheelbase
        self.N = horizon_n
        self.dt = dt
        self.steer_max = steer_max
        self.accel_min, self.accel_max = accel_bounds
        self.steer_rate_max = steer_rate_max
        self._prev_steer = 0.0
        self._build_solver()

    def _build_solver(self) -> None:
        # acados はここでのみ import (未インストール環境を隔離)
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
        import casadi as ca

        # --- モデル定義 ---
        X = ca.SX.sym("X"); Y = ca.SX.sym("Y")
        psi = ca.SX.sym("psi"); v = ca.SX.sym("v")
        x = ca.vertcat(X, Y, psi, v)
        a = ca.SX.sym("a"); delta = ca.SX.sym("delta")
        u = ca.vertcat(a, delta)

        xdot = ca.SX.sym("xdot", 4)
        f_expl = ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v / self.L * ca.tan(delta),
            a,
        )
        model = AcadosModel()
        model.name = "kinematic_bicycle"
        model.x = x
        model.u = u
        model.xdot = xdot
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N * self.dt

        # --- コスト (linear least squares) ---
        # y = [X, Y, psi, v, a, delta] を参照へ追従
        nx, nu = 4, 2
        ny = nx + nu
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        Vx = np.zeros((ny, nx)); Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu)); Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)
        # 重み: 位置・向き重視、速度中、入力は正則化
        ocp.cost.W = np.diag([10.0, 10.0, 5.0, 2.0, 0.1, 0.5])
        ocp.cost.W_e = np.diag([10.0, 10.0, 5.0, 2.0])
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        # --- 制約 ---
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([self.accel_min, -self.steer_max])
        ocp.constraints.ubu = np.array([self.accel_max, self.steer_max])
        ocp.constraints.x0 = np.zeros(nx)

        # --- ソルバ設定 (Real-Time Iteration) ---
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 1

        self._ca = ca
        self._solver = AcadosOcpSolver(ocp, json_file="acados_bicycle.json")
        self._nx, self._nu, self._ny = nx, nu, ny

    def _resample_reference(self, target: PlanningTarget) -> np.ndarray:
        """PlanningTarget.trajectory を MPC の時間格子 (N+1点 @dt) へ補間。

        trajectory 各行 [t, x, y, theta, v, kappa]。
        MPC の各ステージ時刻 k*dt に線形補間して [X,Y,psi,v] を得る。
        """
        traj = target.trajectory
        t_src = traj[:, 0]
        t_dst = np.arange(self.N + 1) * self.dt
        ref = np.zeros((self.N + 1, 4))
        for j, col in enumerate([1, 2, 3, 4]):   # x,y,theta,v
            # theta は連続性のため unwrap してから補間
            src = np.unwrap(traj[:, col]) if col == 3 else traj[:, col]
            ref[:, j] = np.interp(t_dst, t_src, src)
        return ref

    def _solve(self, state: VehicleState, target: PlanningTarget
               ) -> tuple[ControlCommand, MpcStatus]:
        ref = self._resample_reference(target)

        # 初期状態の設定
        x0 = state.as_array()
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # 参照設定
        for k in range(self.N):
            yref = np.concatenate([ref[k], np.zeros(self._nu)])
            # 速度参照から近似加速度参照を作り、a のトラッキング先にする
            self._solver.set(k, "yref", yref)
        self._solver.set(self.N, "yref", ref[self.N])

        # warm start: control_ref(πctrl) を初期入力推定に使う
        if target.has_control_ref:
            a_ref, steer_rate_ref = target.control_ref
            delta_ws = np.clip(state.steer_angle + steer_rate_ref * self.dt,
                               -self.steer_max, self.steer_max)
            for k in range(self.N):
                self._solver.set(k, "u", np.array([a_ref, delta_ws]))

        # 求解 (RTI 1 反復)
        st = self._solver.solve()
        if st != 0:
            raise RuntimeError(f"acados solve failed status={st}")

        u0 = self._solver.get(0, "u")
        accel, steer = float(u0[0]), float(u0[1])

        # 舵角レート制限 (スルーレート)
        max_dsteer = self.steer_rate_max * self.dt
        steer = float(np.clip(steer, self._prev_steer - max_dsteer,
                              self._prev_steer + max_dsteer))
        self._prev_steer = steer

        # 追従誤差 (0 ステージの参照との差)
        err_lat = float(np.hypot(ref[0, 0] - state.x, ref[0, 1] - state.y))

        cmd = ControlCommand(stamp_ns=now_ns(), plan_id=target.plan_id,
                             steer_angle=steer, accel=accel)
        status = MpcStatus(stamp_ns=now_ns(), solve_ok=True, solve_time_ms=0.0,
                           tracking_err_lat=err_lat,
                           fallback=FallbackState.NONE,
                           cost=float(self._solver.get_cost()))
        return cmd, status


# ---------------------------------------------------------------------------
# フォールバック実装: acados 不在でも配線検証できる純 numpy 版
# ---------------------------------------------------------------------------
class SimpleMpcController(MpcControllerBase):
    """acados が無い環境用の軽量 pure-pursuit + P 制御。

    厳密な MPC ではないが、契約(update_plan/step/フォールバック)を満たし、
    E2E 側インターフェースや RPC 配線の検証に使える。
    """

    def __init__(self, wheelbase=2.7, dt=0.1, steer_max=0.6,
                 lookahead=5.0, kp_speed=1.0, **kw):
        super().__init__(**kw)
        self.L = wheelbase
        self.dt = dt
        self.steer_max = steer_max
        self.lookahead = lookahead
        self.kp_speed = kp_speed

    def _solve(self, state: VehicleState, target: PlanningTarget):
        traj = target.trajectory
        # lookahead 距離に最も近い参照点を選ぶ (pure pursuit)
        dists = np.hypot(traj[:, 1] - state.x, traj[:, 2] - state.y)
        idx = int(np.argmin(np.abs(dists - self.lookahead)))
        tx, ty = traj[idx, 1], traj[idx, 2]

        # ego 座標系での目標点角度
        dx = (tx - state.x) * np.cos(-state.theta) - (ty - state.y) * np.sin(-state.theta)
        dy = (tx - state.x) * np.sin(-state.theta) + (ty - state.y) * np.cos(-state.theta)
        ld = max(np.hypot(dx, dy), 1e-3)
        # pure pursuit の曲率 -> 舵角
        curvature = 2.0 * dy / (ld ** 2)
        steer = float(np.clip(np.arctan(self.L * curvature),
                              -self.steer_max, self.steer_max))

        # 速度追従 (P 制御)
        v_ref = float(traj[idx, 4])
        accel = float(np.clip(self.kp_speed * (v_ref - state.v), -5.0, 3.0))

        # control_ref があれば加速度を混合 (πctrl 反映)
        if target.has_control_ref:
            accel = 0.5 * accel + 0.5 * float(target.control_ref[0])

        cmd = ControlCommand(stamp_ns=now_ns(), plan_id=target.plan_id,
                             steer_angle=steer, accel=accel)
        status = MpcStatus(stamp_ns=now_ns(), solve_ok=True, solve_time_ms=0.0,
                           tracking_err_lat=float(dists[idx]),
                           fallback=FallbackState.NONE)
        return cmd, status
