# e2e_mpc_bridge

E2E Planner ↔ MPC のデータ契約 (レポート第5章) に基づく、Acados Python wrapper
(`MpcControllerBase` 実装) と E2E 側インターフェースの実装。

## 構成

| パス | 役割 |
|---|---|
| `mpc/contract.py` | データ契約: PlanningTarget / VehicleState / ControlCommand / MpcStatus |
| `mpc/mpc_base.py` | `MpcControllerBase` 抽象基底。スレッド安全な update_plan/step、失効検出、フォールバック |
| `mpc/acados_mpc.py` | `AcadosMpcController` (bicycle OCP, SQP-RTI) と `SimpleMpcController` (acados不在時) |
| `e2e/e2e_interface.py` | `E2EPlannerBase` と UniAD/VAD・Hydra-NeXt アダプタ、E2EMpcRunner |
| `examples/demo_e2e_mpc.py` | 契約・πctrl・失効・EMERGENCY・スレッド安全性の検証デモ |

## 設計の要点

1. **レート分離のスレッド安全性**: `update_plan`(10-20Hz, 非同期) と
   `step`(50-100Hz, 同期) を Lock で保護。ロック保持は参照差し替えのみに限定し、
   最適化計算はロック外で実行。
2. **失効検出**: `step` 内で `state.stamp_ns - target.stamp_ns > plan_timeout_s`
   なら STALE_PLAN として緊急減速へフォールバック。
3. **πctrl の活用**: Hydra-NeXt の control_ref を Acados の warm start(初期入力)に
   使い、MPC が運動学的実行可能性を保証する初期解として活かす(πdp 相当)。
4. **バージョン/依存の隔離**: acados は `_build_solver` 内でのみ遅延 import。
   `SimpleMpcController` は acados 非依存で、実 MPC 導入前の配線検証に使える。

## 使い方

```python
from acados_mpc import AcadosMpcController      # or SimpleMpcController
from e2e_interface import HydraNextPlanner, E2EMpcRunner

mpc = AcadosMpcController(wheelbase=2.7, horizon_n=20, dt=0.1,
                         plan_timeout_s=0.5)
planner = HydraNextPlanner(forward_fn=my_model_forward, dt=0.5)
runner = E2EMpcRunner(planner, mpc)

# 低レートスレッド:  runner.on_sensor(sensor_batch)  -> mpc.update_plan
# 高レートスレッド:  cmd, status = runner.tick(vehicle_state)
```

## 実 acados での実行

`SimpleMpcController` はどこでも動く。`AcadosMpcController` は acados 本体
(C ライブラリ) と acados_template / casadi が必要:

```bash
# acados 本体をソースビルド後
pip install casadi
export ACADOS_SOURCE_DIR=/path/to/acados
pip install -e /path/to/acados/interfaces/acados_template
python examples/demo_e2e_mpc.py   # AcadosMpcController も利用可能に
```

デモは acados 不在なら自動で SimpleMpcController にフォールバックする。

## E2E 側を差し替えても MPC は不変

`E2EPlannerBase` のサブクラスを増やすだけ:
- `TrajectoryOnlyPlanner`: UniAD/VAD (軌跡のみ、control_ref なし)
- `HydraNextPlanner`: Hydra-NeXt (πtraj + πctrl -> control_ref)

出力が PlanningTarget に正規化される限り、MPC 側 (`MpcControllerBase`) は
一切変更不要。前段の BEVFormer ブリッジや評価ハーネスと同じ「境界固定」の思想。
"""
