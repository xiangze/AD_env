"""
tools/compression/distill.py
知識蒸留スクリプト (UniAD / VAD)

Teacher (FP32 大モデル) → Student (軽量モデル) への蒸留。
蒸留の種類:
    response  : 最終出力の KL 蒸留 (最もシンプル)
    feature   : 中間 BEV 特徴量の L2 蒸留
    combined  : response + feature の加重和

対応パターン:
    UniAD R101 → R50     (バックボーン蒸留)
    VAD-Base   → VAD-Tiny (同一アーキテクチャ、容量削減)

使い方:
    # UniAD R101 (teacher) → R50 (student) 蒸留
    python tools/compression/distill.py \
        --teacher-config  projects/configs/stage2_e2e/base_e2e.py \
        --teacher-ckpt    ckpts/uniad_base_e2e.pth \
        --student-config  projects/configs/stage2_e2e/base_e2e_r50.py \
        --student-ckpt    ckpts/r50_backbone.pth \
        --mode     response \
        --gpus     8 \
        --epochs   10 \
        --lr       1e-4 \
        --out-dir  work_dirs/distill_r101_to_r50

    # VAD-Base → VAD-Tiny 蒸留
    python tools/compression/distill.py \
        --model    vad \
        --teacher-config  projects/configs/VAD/VAD_base_stage_2.py \
        --teacher-ckpt    ckpts/VAD_base.pth \
        --student-config  projects/configs/VAD/VAD_tiny_stage_2.py \
        --mode     response \
        --gpus     4 \
        --epochs   20 \
        --out-dir  work_dirs/distill_vad_base_to_tiny
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="知識蒸留スクリプト")
    p.add_argument("--model",           default="uniad",
                   choices=["uniad", "vad"])
    p.add_argument("--teacher-config",  required=True)
    p.add_argument("--teacher-ckpt",    required=True)
    p.add_argument("--student-config",  required=True)
    p.add_argument("--student-ckpt",    default="",
                   help="Student の初期化ckpt (なければランダム初期化)")
    p.add_argument("--mode",            default="response",
                   choices=["response", "feature", "combined"],
                   help="蒸留モード")
    p.add_argument("--temperature",     type=float, default=4.0,
                   help="KL 蒸留の温度パラメータ")
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="蒸留損失の重み (1-alpha が通常損失)")
    p.add_argument("--feat-loss-weight",type=float, default=1.0,
                   help="feature 蒸留損失の重み (combined モードのみ)")
    p.add_argument("--gpus",            type=int,   default=8)
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--batch-size",      type=int,   default=1)
    p.add_argument("--out-dir",         required=True)
    p.add_argument("--data",            default="",
                   help="カスタムデータの pkl パス")
    p.add_argument("--dry-run",         action="store_true")
    return p.parse_args()


# ── 蒸留損失の実装 ───────────────────────────────────────────
class ResponseDistillLoss(nn.Module):
    """
    最終出力の KL ダイバージェンス蒸留損失。
    Planning / Motion の確率出力に適用する。

    Args:
        temperature (float): 高いほど教師の分布が滑らかになる
        alpha (float):       蒸留損失の重み
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels:    torch.Tensor | None = None,
        task_loss:      torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # KL 蒸留損失
        p_t = F.softmax(teacher_logits / self.T, dim=-1)
        p_s = F.log_softmax(student_logits / self.T, dim=-1)
        kl  = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)

        total = self.alpha * kl
        if task_loss is not None:
            total = total + (1 - self.alpha) * task_loss

        return {"distill_kl": kl, "total": total}


class FeatureDistillLoss(nn.Module):
    """
    中間 BEV 特徴量の L2 蒸留損失。
    Teacher と Student の BEV 特徴次元が異なる場合は
    Adapter (1×1 conv) で合わせる。

    Args:
        teacher_dim (int): Teacher の BEV 特徴次元
        student_dim (int): Student の BEV 特徴次元
    """

    def __init__(self, teacher_dim: int = 256, student_dim: int = 256):
        super().__init__()
        self.adapter: nn.Module | None = None
        if teacher_dim != student_dim:
            self.adapter = nn.Conv2d(
                student_dim, teacher_dim, kernel_size=1, bias=False
            )

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.adapter is not None:
            student_feat = self.adapter(student_feat)

        # Teacher の特徴は勾配を流さない
        teacher_feat = teacher_feat.detach()
        l2 = F.mse_loss(student_feat, teacher_feat)
        return {"distill_feat_l2": l2, "total": l2}


class CombinedDistillLoss(nn.Module):
    """ResponseDistillLoss + FeatureDistillLoss の組み合わせ。"""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feat_weight: float = 1.0,
        teacher_dim: int = 256,
        student_dim: int = 256,
    ):
        super().__init__()
        self.resp = ResponseDistillLoss(temperature, alpha)
        self.feat = FeatureDistillLoss(teacher_dim, student_dim)
        self.feat_weight = feat_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feat:   torch.Tensor,
        teacher_feat:   torch.Tensor,
        task_loss:      torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        r = self.resp(student_logits, teacher_logits, task_loss=task_loss)
        f = self.feat(student_feat, teacher_feat)
        total = r["total"] + self.feat_weight * f["distill_feat_l2"]
        return {**r, **f, "total": total}


# ── 蒸留 Config 生成 ─────────────────────────────────────────
def build_distill_config(args: argparse.Namespace) -> str:
    """蒸留用の mmdet3d config を動的に生成する。"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_section = ""
    if args.data:
        data_section = f"""
data = dict(
    train=dict(ann_file='{args.data}', data_root=''),
)
"""

    content = f"""\
# Auto-generated distillation config
# Teacher: {args.teacher_config}
# Student: {args.student_config}
# Mode   : {args.mode}
_base_ = ['{args.student_config}']

# ── 蒸留パラメータ ──────────────────────────────────────────
distill = dict(
    teacher_config = '{args.teacher_config}',
    teacher_ckpt   = '{args.teacher_ckpt}',
    mode           = '{args.mode}',
    temperature    = {args.temperature},
    alpha          = {args.alpha},
    feat_loss_weight = {args.feat_loss_weight},
)

# ── 最適化設定 (蒸留用: 低 LR) ──────────────────────────────
optimizer = dict(
    type='AdamW',
    lr={args.lr},
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={{
            'img_backbone': dict(lr_mult=0.1),
            'planning_head': dict(lr_mult=2.0),
        }}
    ),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs={args.epochs})
evaluation = dict(interval={max(1, args.epochs // 4)})
checkpoint_config = dict(interval={max(1, args.epochs // 4)})
fp16 = dict(loss_scale='dynamic')
{data_section}
"""
    cfg_path = str(out_dir / "distill_config.py")
    Path(cfg_path).write_text(content)
    return cfg_path


# ── 蒸留ジョブの起動 ─────────────────────────────────────────
def launch_distillation(
    args: argparse.Namespace,
    distill_config: str,
) -> None:
    """蒸留学習を dist_train 経由で起動する。"""
    root = "." if args.model == "uniad" else "/workspace/VAD"
    train_script = (
        "tools/uniad_dist_train.sh" if args.model == "uniad"
        else "tools/train.py"
    )

    if args.model == "uniad":
        cmd = [
            "bash", train_script,
            distill_config,
            str(args.gpus),
            "--work-dir", args.out_dir,
        ]
        if args.student_ckpt:
            cmd += ["--load-from", args.student_ckpt]
    else:
        # VAD: torch.distributed.run 経由
        cmd = [
            "python3", "-m", "torch.distributed.run",
            f"--nproc_per_node={args.gpus}",
            "--master_port=2333",
            "tools/train.py",
            distill_config,
            "--launcher", "pytorch",
            "--deterministic",
            "--work-dir", args.out_dir,
        ]
        if args.student_ckpt:
            cmd += ["--load-from", args.student_ckpt]

    print(f"\n[distill] 蒸留学習起動:")
    print("  " + " \\\n  ".join(cmd))

    if not args.dry_run:
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            sys.exit(result.returncode)


def main() -> None:
    args = parse_args()

    print("=" * 56)
    print(f"  知識蒸留 ({args.model.upper()})")
    print(f"  Teacher: {args.teacher_config}")
    print(f"  Student: {args.student_config}")
    print(f"  Mode   : {args.mode}")
    print(f"  α      : {args.alpha} (蒸留損失の重み)")
    print(f"  T      : {args.temperature} (蒸留温度)")
    print(f"  GPUs   : {args.gpus}  Epochs: {args.epochs}")
    print("=" * 56)

    # 1. 蒸留 config 生成
    distill_config = build_distill_config(args)
    print(f"\n[distill] 生成 config: {distill_config}")

    # 2. 蒸留学習起動
    launch_distillation(args, distill_config)

    # 3. 完了後に精度評価を自動実行するか案内
    student_ckpt = str(
        Path(args.out_dir) /
        f"epoch_{args.epochs}.pth"
    )
    print(f"\n[distill] 完了後の精度評価:")
    print(f"  python eval_scripts/eval_compressed.py \\")
    print(f"      --model   {args.model} \\")
    print(f"      --config  {args.student_config} \\")
    print(f"      --ckpts   {args.teacher_ckpt}:fp32 \\")
    print(f"               {student_ckpt}:distilled \\")
    print(f"      --tasks   planning \\")
    print(f"      --out-dir {args.out_dir}/eval")


if __name__ == "__main__":
    main()
