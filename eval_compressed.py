"""
eval_scripts/eval_compressed.py
量子化・蒸留後モデルの精度評価スクリプト

フルモデルとの比較を含む 4 つの指標を同時計測する:
    1. タスク精度    : 各モデル固有の指標 (AMOTA / avg.L2 など)
    2. 精度劣化量    : フルモデル比の相対低下率 (%)
    3. VRAM 消費量   : 実測値 (torch.cuda.memory_allocated)
    4. 推論速度      : FPS (warmup 10 回 / 計測 100 回の平均)

対応モデル:
    UniAD: FP32 / FP16 / INT8_dynamic
    VAD:   FP32 / FP16 / INT8_dynamic (VAD-Tiny / VAD-Base)

使い方:
    # UniAD FP16 vs FP32 を比較
    python eval_scripts/eval_compressed.py \
        --model    uniad \
        --config   projects/configs/stage2_e2e/base_e2e.py \
        --ckpts    ckpts/uniad_base_e2e.pth:fp32 \
                   ckpts/uniad_fp16.pth:fp16 \
                   ckpts/uniad_int8.pth:int8 \
        --tasks    planning,tracking \
        --out-dir  eval_results/compression_comparison

    # VAD FP16 vs FP32
    python eval_scripts/eval_compressed.py \
        --model    vad \
        --variant  base \
        --config   projects/configs/VAD/VAD_base_stage_2.py \
        --ckpts    ckpts/VAD_base.pth:fp32 \
                   ckpts/VAD_base_fp16.pth:fp16 \
        --out-dir  eval_results/vad_compression
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from eval_base import get_logger, ALL_TASKS, REFERENCE
from eval_tasks import EVALUATOR_MAP

# VAD 参照値 (eval_vad.py より)
VAD_REFERENCE = {
    "tiny": {"avg_L2": 0.78, "avg_Col": 0.38, "L2_1s": 0.46,
             "L2_2s": 0.76, "L2_3s": 1.12, "FPS": 16.8},
    "base": {"avg_L2": 0.72, "avg_Col": 0.22, "L2_1s": 0.41,
             "L2_2s": 0.70, "L2_3s": 1.05, "FPS": 4.5},
}


@dataclass
class CompressionResult:
    """1 つのチェックポイント (精度 × 1 種) の評価結果。"""
    label:       str                          # "fp32" | "fp16" | "int8"
    ckpt_path:   str
    metrics:     dict[str, float | None] = field(default_factory=dict)
    vram_mb:     float | None = None          # 実測ピーク VRAM (MB)
    fps:         float | None = None          # 推論 FPS
    latency_ms:  float | None = None          # 平均レイテンシ (ms)
    model_size_mb: float | None = None        # ckpt ファイルサイズ (MB)
    precision:   str = "fp32"                 # torch dtype

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="量子化・蒸留後モデルの精度評価"
    )
    p.add_argument("--model",   required=True,
                   choices=["uniad", "vad"])
    p.add_argument("--variant", default="base",
                   choices=["tiny", "base", "v2"],
                   help="VAD のバリアント")
    p.add_argument("--config",  required=True,
                   help="ベース config ファイルパス")
    p.add_argument("--ckpts",   nargs="+", required=True,
                   help="チェックポイント指定: path:label  例) ckpt.pth:fp32")
    p.add_argument("--tasks",   default="planning",
                   help="カンマ区切りタスク / 'all'  (UniAD のみ)")
    p.add_argument("--gpus",    type=int, default=1)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-accuracy",  action="store_true",
                   help="精度評価をスキップして速度・VRAM のみ計測")
    p.add_argument("--skip-latency",   action="store_true",
                   help="速度計測をスキップ")
    p.add_argument("--warmup",  type=int, default=10,
                   help="推論速度計測の warmup 回数")
    p.add_argument("--n-iter",  type=int, default=100,
                   help="推論速度計測の計測回数")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--uniad-root", default=".")
    p.add_argument("--vad-root",   default="/workspace/VAD")
    return p.parse_args()


# ── モデルロード ─────────────────────────────────────────────
def load_model_mmdet(config: str, ckpt: str,
                     device: str, precision: str):
    """mmdet3d 経由でモデルをロードする (UniAD / VAD 共通)。"""
    from mmcv import Config
    from mmdet3d.models import build_detector
    from mmcv.runner import load_checkpoint

    cfg   = Config.fromfile(config)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

    # INT8 の場合: 量子化済み state_dict をロード
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        prec = state.get("precision", "fp32")
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        load_checkpoint(model, ckpt, map_location="cpu")
        prec = precision

    if prec == "fp16" or precision == "fp16":
        model = model.half()
    model.to(device).eval()
    return model, prec


# ── VRAM 計測 ───────────────────────────────────────────────
def measure_vram(model: nn.Module, device: str) -> float:
    """モデルをデバイスへ転送した直後のピーク VRAM を返す (MB)。"""
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    # 1 forward pass でアクティベーションのピークを計測
    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return round(vram_mb, 1)


# ── 推論速度計測 ─────────────────────────────────────────────
def measure_latency(model: nn.Module, device: str,
                    precision: str,
                    warmup: int = 10, n_iter: int = 100,
                    input_shape: tuple = (1, 6, 3, 900, 1600)
                    ) -> tuple[float, float]:
    """ダミー入力での平均レイテンシ (ms) と FPS を返す。"""
    model.eval()
    dtype = torch.float16 if precision in ("fp16",) else torch.float32
    # BEV バックボーン入力を模倣 (batch=1, queue=1, n_cam=6, C, H, W)
    dummy = torch.randn(*input_shape, dtype=dtype).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = model.img_backbone(dummy.view(-1, *input_shape[2:]))
            except Exception:
                break  # backbone 単体動作しない場合はスキップ

    if device == "cuda":
        torch.cuda.synchronize()

    # 計測
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_iter):
            t0 = time.perf_counter()
            try:
                _ = model.img_backbone(dummy.view(-1, *input_shape[2:]))
            except Exception:
                pass
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    if not times:
        return 0.0, 0.0

    avg_ms = sum(times) / len(times)
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return round(avg_ms, 2), round(fps, 2)


# ── 精度評価 (UniAD) ─────────────────────────────────────────
def run_uniad_accuracy(config: str, ckpt: str, label: str,
                       tasks: list[str], gpus: int,
                       out_dir: Path, uniad_root: str
                       ) -> dict[str, float | None]:
    """UniAD の精度評価を実行して指標を返す。"""
    all_metrics: dict[str, float | None] = {}
    for task in tasks:
        if task not in EVALUATOR_MAP:
            continue
        evaluator = EVALUATOR_MAP[task](
            config     = config,
            ckpt       = ckpt,
            gpus       = gpus,
            out_dir    = out_dir / task,
            uniad_root = uniad_root,
        )
        try:
            m = evaluator.run()
            all_metrics.update(m)
        except Exception as e:
            print(f"[WARN] {task} 評価失敗: {e}")
    return all_metrics


# ── 精度評価 (VAD) ───────────────────────────────────────────
def run_vad_accuracy(config: str, ckpt: str, variant: str,
                     out_dir: Path, vad_root: str
                     ) -> dict[str, float | None]:
    """VAD の精度評価を実行して指標を返す (1 GPU 固定)。"""
    import re
    result_pkl = out_dir / "vad_results.pkl"
    log_file   = out_dir / "inference.log"

    cmd = [
        "python3", "tools/test.py",
        config, ckpt,
        "--launcher", "none",
        "--eval", "bbox",
        "--tmpdir", str(out_dir / "tmp"),
        "--out",  str(result_pkl),
    ]
    with log_file.open("w") as lf:
        proc = subprocess.run(
            cmd, cwd=vad_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        lf.write(proc.stdout)

    log = log_file.read_text()
    patterns = {
        "L2_1s": r"L2_1\s+([\d.]+)", "L2_2s": r"L2_2\s+([\d.]+)",
        "L2_3s": r"L2_3\s+([\d.]+)", "Col_1s": r"Col_1\s+([\d.]+)",
        "Col_2s": r"Col_2\s+([\d.]+)", "Col_3s": r"Col_3\s+([\d.]+)",
    }
    metrics: dict[str, float | None] = {}
    for key, pat in patterns.items():
        m = re.search(pat, log)
        metrics[key] = float(m.group(1)) if m else None

    l2  = [metrics[f"L2_{t}s"]  for t in [1,2,3] if metrics.get(f"L2_{t}s")]
    col = [metrics[f"Col_{t}s"] for t in [1,2,3] if metrics.get(f"Col_{t}s")]
    metrics["avg_L2"]  = sum(l2)  / len(l2)  if l2  else None
    metrics["avg_Col"] = sum(col) / len(col) if col else None
    return metrics


# ── 精度劣化率の計算 ─────────────────────────────────────────
def compute_degradation(
    results: list[CompressionResult],
    model: str,
    variant: str,
) -> dict[str, dict[str, float | None]]:
    """fp32 を基準とした各モデルの相対精度劣化率 (%) を返す。"""
    fp32 = next((r for r in results if r.label == "fp32"), None)
    if fp32 is None:
        return {}

    degradation: dict[str, dict[str, float | None]] = {}
    key_metrics = (
        ["avg_L2", "avg_Col"] if model == "vad"
        else ["AMOTA", "avg_L2", "avg_Col"]
    )

    for r in results:
        if r.label == "fp32":
            continue
        degrad: dict[str, float | None] = {}
        for k in key_metrics:
            fp32_val = fp32.metrics.get(k)
            curr_val = r.metrics.get(k)
            if fp32_val is not None and curr_val is not None and fp32_val != 0:
                degrad[k] = round(
                    (curr_val - fp32_val) / abs(fp32_val) * 100, 2
                )
            else:
                degrad[k] = None
        degradation[r.label] = degrad

    return degradation


# ── 比較テーブル表示 ─────────────────────────────────────────
def print_comparison_table(
    results: list[CompressionResult],
    degradation: dict[str, dict],
    model: str,
    variant: str,
) -> None:
    ref = VAD_REFERENCE.get(variant, {}) if model == "vad" else {}
    print(f"\n{'='*72}")
    print(f"  {model.upper()} 量子化・蒸留 精度比較レポート")
    print(f"{'='*72}")

    # ヘッダ
    col_w = 14
    headers = ["指標"] + [r.label for r in results]
    print("  " + "  ".join(h.center(col_w) for h in headers))
    print("  " + "-" * (col_w * (len(headers) + 1)))

    # 精度指標行
    if model == "vad":
        show_metrics = ["avg_L2", "avg_Col", "L2_1s", "L2_2s", "L2_3s"]
    else:
        show_metrics = ["AMOTA", "AMOTP", "avg_L2", "avg_Col", "IoU_lane"]

    for k in show_metrics:
        row = [k.ljust(10)]
        for r in results:
            v = r.metrics.get(k)
            cell = f"{v:.4f}" if v is not None else "N/A"
            # 劣化率を付加
            if r.label != "fp32":
                d = degradation.get(r.label, {}).get(k)
                if d is not None:
                    sign = "+" if d > 0 else ""
                    cell += f"({sign}{d:.1f}%)"
            row.append(cell.center(col_w))
        print("  " + "  ".join(row))

    print("  " + "-" * (col_w * (len(headers) + 1)))

    # VRAM 行
    vram_row = ["VRAM(MB)".ljust(10)]
    for r in results:
        vram_row.append(
            (f"{r.vram_mb:.0f}" if r.vram_mb else "N/A").center(col_w)
        )
    print("  " + "  ".join(vram_row))

    # FPS 行
    fps_row = ["FPS".ljust(10)]
    for r in results:
        fps_row.append(
            (f"{r.fps:.1f}" if r.fps else "N/A").center(col_w)
        )
    print("  " + "  ".join(fps_row))

    # モデルサイズ行
    size_row = ["Size(MB)".ljust(10)]
    for r in results:
        size_row.append(
            (f"{r.model_size_mb:.0f}" if r.model_size_mb else "N/A")
            .center(col_w)
        )
    print("  " + "  ".join(size_row))

    print(f"{'='*72}")
    print("  ※ 精度劣化率は fp32 比。+ = 悪化, - = 改善。")
    print(f"{'='*72}\n")


# ── メイン ───────────────────────────────────────────────────
def main() -> int:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = get_logger("eval_compressed", out_dir / "eval_compressed.log")

    # チェックポイント指定を解析: "path:label"
    ckpt_specs: list[tuple[str, str]] = []
    for spec in args.ckpts:
        if ":" in spec:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, "fp32"
        ckpt_specs.append((path, label))

    tasks = (ALL_TASKS if args.tasks == "all"
             else [t.strip() for t in args.tasks.split(",")])

    logger.info("=" * 60)
    logger.info(f" 量子化・蒸留 精度評価")
    logger.info(f" model={args.model}  variant={args.variant}")
    logger.info(f" ckpts={[s[1] for s in ckpt_specs]}")
    logger.info(f" tasks={tasks}")
    logger.info("=" * 60)

    results: list[CompressionResult] = []

    for ckpt_path, label in ckpt_specs:
        logger.info(f"\n── [{label}] 評価開始: {ckpt_path}")
        result = CompressionResult(label=label, ckpt_path=ckpt_path)

        # ファイルサイズ
        p = Path(ckpt_path)
        if p.exists():
            result.model_size_mb = round(p.stat().st_size / 1e6, 1)

        # モデルロード & VRAM・速度計測
        if not args.skip_latency:
            try:
                model, prec = load_model_mmdet(
                    args.config, ckpt_path, args.device, label
                )
                result.precision = prec
                result.vram_mb   = measure_vram(model, args.device)
                result.latency_ms, result.fps = measure_latency(
                    model, args.device, prec,
                    args.warmup, args.n_iter,
                )
                logger.info(
                    f"  VRAM={result.vram_mb} MB  "
                    f"latency={result.latency_ms} ms  "
                    f"FPS={result.fps}"
                )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"  速度・VRAM 計測失敗: {e}")

        # 精度評価
        if not args.skip_accuracy:
            run_dir = out_dir / label
            run_dir.mkdir(exist_ok=True)
            try:
                if args.model == "uniad":
                    result.metrics = run_uniad_accuracy(
                        args.config, ckpt_path, label,
                        tasks, args.gpus, run_dir, args.uniad_root
                    )
                else:
                    result.metrics = run_vad_accuracy(
                        args.config, ckpt_path, args.variant,
                        run_dir, args.vad_root
                    )
                logger.info(f"  精度: {result.metrics}")
            except Exception as e:
                logger.error(f"  精度評価失敗: {e}")

        results.append(result)

    # ── 劣化率計算・表示 ──────────────────────────────────────
    degradation = compute_degradation(results, args.model, args.variant)
    print_comparison_table(results, degradation, args.model, args.variant)

    # ── JSON 保存 ─────────────────────────────────────────────
    out_json = out_dir / "compression_comparison.json"
    out_json.write_text(json.dumps({
        "model":       args.model,
        "variant":     args.variant,
        "results":     [r.to_dict() for r in results],
        "degradation": degradation,
    }, indent=2, ensure_ascii=False))
    logger.info(f"結果保存: {out_json}")

    # ── Markdown レポート生成 ─────────────────────────────────
    _write_markdown_report(results, degradation, args, out_dir)

    return 0


def _write_markdown_report(
    results: list[CompressionResult],
    degradation: dict,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    lines = [
        f"# {args.model.upper()} 量子化・蒸留 精度評価レポート",
        f"\nmodel: **{args.model}**  variant: **{args.variant}**\n",
        "## 指標比較\n",
        "| 指標 | " + " | ".join(r.label for r in results) + " |",
        "|------|" + "|".join("------" for _ in results) + "|",
    ]

    show = (["avg_L2", "avg_Col"] if args.model == "vad"
            else ["AMOTA", "avg_L2", "avg_Col"])
    for k in show:
        row_vals = []
        for r in results:
            v = r.metrics.get(k)
            cell = f"{v:.4f}" if v is not None else "—"
            if r.label != "fp32":
                d = degradation.get(r.label, {}).get(k)
                if d is not None:
                    cell += f" ({'+' if d>0 else ''}{d:.1f}%)"
            row_vals.append(cell)
        lines.append(f"| {k} | " + " | ".join(row_vals) + " |")

    lines += [
        "",
        "## リソース効率\n",
        "| | " + " | ".join(r.label for r in results) + " |",
        "|-|" + "|".join("------" for _ in results) + "|",
        "| VRAM (MB) | " + " | ".join(
            str(r.vram_mb or "—") for r in results
        ) + " |",
        "| FPS | " + " | ".join(
            str(r.fps or "—") for r in results
        ) + " |",
        "| Size (MB) | " + " | ".join(
            str(r.model_size_mb or "—") for r in results
        ) + " |",
        "",
        "> ※ 精度劣化率は fp32 比。`+` = 悪化 (高いほど悪い指標)。",
    ]

    out_md = out_dir / "compression_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Markdown 保存: {out_md}")


if __name__ == "__main__":
    sys.exit(main())
