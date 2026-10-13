"""
lqr_tracker.py — E2E プランナ軌道の追従 LQR(acados 不要・numpy/scipy のみ)

acados MPC の軽量代替。拘束は扱えないが、C コード生成やビルドが不要で即実行できる。
運動学自転車モデルを参照点まわりで線形化し、各ステップで離散時間 Riccati(DARE)を
解く時変 LQR として制御量 [a, delta] を求める。

状態 x = [X, Y, psi, v]、制御 u = [a, delta]、ホイールベース L。
連続時間ダイナミクス:
    Xdot   = v cos psi
    Ydot   = v sin psi
    psidot = v / L * tan(delta)
    vdot   = a

インターフェースは uniad_vad_acados_mpc と揃える:
    build_tracker(...) -> tracker
    plan_to_reference(...)   # acados 版と同一(こちらにも用意)
    step(tracker, x0, ref, N) -> (u0=[a,delta], status)
"""
import numpy as np
from scipy.linalg import solve_discrete_are


class LQRTracker:
    def __init__(self, L=2.9, Tf=3.0, N=20,
                 Q=None, R=None,
                 a_bounds=(-4.0, 2.0), delta_bounds=(-0.6, 0.6)):
        self.L = L
        self.dt = Tf / N
        # 状態重み [X, Y, psi, v]、制御重み [a, delta]
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0]) if Q is None else np.asarray(Q, float)
        self.R = np.diag([0.1, 0.1]) if R is None else np.asarray(R, float)
        self.a_lo, self.a_hi = a_bounds
        self.d_lo, self.d_hi = delta_bounds

    # --- 参照点まわりの連続ヤコビアン ---
    def _jacobians(self, psi, v, delta):
        L = self.L
        A = np.array([
            [0.0, 0.0, -v * np.sin(psi), np.cos(psi)],
            [0.0, 0.0,  v * np.cos(psi), np.sin(psi)],
            [0.0, 0.0,  0.0,             np.tan(delta) / L],
            [0.0, 0.0,  0.0,             0.0],
        ])
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, v / (L * np.cos(delta) ** 2)],
            [1.0, 0.0],
        ])
        return A, B

    # --- 参照制御(reference feedforward)を軌道から推定 ---
    def _reference_input(self, ref):
        """ref: (N+1,4) [X,Y,psi,v] → u_ref: (N,2) [a_ref, delta_ref]"""
        dt, L = self.dt, self.L
        u_ref = np.zeros((len(ref) - 1, 2))
        for k in range(len(ref) - 1):
            v = max(ref[k, 3], 1e-3)
            a_ref = (ref[k + 1, 3] - ref[k, 3]) / dt
            dpsi = np.arctan2(np.sin(ref[k + 1, 2] - ref[k, 2]),
                              np.cos(ref[k + 1, 2] - ref[k, 2]))
            psidot = dpsi / dt
            delta_ref = np.arctan(L * psidot / v)
            u_ref[k] = [a_ref, np.clip(delta_ref, self.d_lo, self.d_hi)]
        return u_ref

    def solve_first(self, x0, ref):
        """現在状態 x0 と参照 ref から、最初の制御 u0=[a,delta] を返す。"""
        u_ref = self._reference_input(ref)
        # 先頭ステップの参照点で線形化(時変 LQR の最初のゲイン)
        psi_r, v_r = ref[0, 2], ref[0, 3]
        delta_r = u_ref[0, 1]
        A, B = self._jacobians(psi_r, v_r, delta_r)
        Ad = np.eye(4) + A * self.dt           # オイラー離散化
        Bd = B * self.dt
        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        K = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

        # 誤差状態(角度は wrap)
        e = x0 - ref[0]
        e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))
        u = u_ref[0] - K @ e                    # u = u_ref - K e
        a = float(np.clip(u[0], self.a_lo, self.a_hi))
        delta = float(np.clip(u[1], self.d_lo, self.d_hi))
        status = 0                              # DARE が解ければ 0
        return np.array([a, delta]), status


# --- acados 版と同一シグネチャのラッパ(run_cosim から共通に呼べるように) ---
def build_tracker(N=20, Tf=3.0, L=2.9):
    return LQRTracker(L=L, Tf=Tf, N=N)


def plan_to_reference(waypoints_xy, N, Tf, v_current):
    """acados 版と同一(時間再サンプリング + 向き・速度の導出)。"""
    M = waypoints_xy.shape[0]
    t_plan = np.linspace(0.0, Tf, M)
    t_mpc = np.linspace(0.0, Tf, N + 1)
    X = np.interp(t_mpc, t_plan, waypoints_xy[:, 0])
    Y = np.interp(t_mpc, t_plan, waypoints_xy[:, 1])
    dX = np.gradient(X, t_mpc)
    dY = np.gradient(Y, t_mpc)
    psi = np.arctan2(dY, dX)
    v = np.sqrt(dX ** 2 + dY ** 2)
    v[0] = v_current
    return np.stack([X, Y, psi, v], axis=1)


def step(tracker, x0, ref, N):
    """uniad_vad_acados_mpc.step と同じ戻り値 (u0, status)。"""
    return tracker.solve_first(x0, ref)


if __name__ == "__main__":
    # 単体スモークテスト
    N, Tf = 20, 3.0
    tr = build_tracker(N, Tf)
    wp = np.array([[2, 0.0], [4, 0.1], [6, 0.3], [8, 0.6], [10, 1.0], [12, 1.5]], float)
    x0 = np.array([0.0, 0.0, 0.0, 8.0])
    ref = plan_to_reference(wp, N, Tf, v_current=x0[3])
    u0, st = step(tr, x0, ref, N)
    print(f"LQR status={st}  a={u0[0]:.3f}  delta={u0[1]:.4f}")
