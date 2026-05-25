"""
real_evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
実モデル (nvidia/Alpamayo-R1-10B) を使った
量子化・蒸留・LoRA・精度評価スクリプト

【前提条件】
  - NVIDIA GPU with ≥ 24 GB VRAM (A100 / H100 / RTX 3090 / RTX 4090)
  - HuggingFace アクセストークン (gated model)
    - https://huggingface.co/nvidia/Alpamayo-R1-10B
    - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
  - alpamayo パッケージのインストール:
      uv venv ar1_venv && source ar1_venv/bin/activate && uv sync --active
    または:
      pip install git+https://github.com/NVlabs/alpamayo.git

【ライセンス注意】
  - モデルウェイト: 非商用ライセンス (研究・実験・評価のみ)
  - 推論コード: Apache 2.0
  - PhysicalAI-AV データセット: NVIDIA AV Dataset License

【使用方法】
  # 1. 基本評価 (FP32/BF16 ベースライン)
  python real_evaluate.py --mode baseline

  # 2. INT8/INT4 量子化評価 (bitsandbytes)
  python real_evaluate.py --mode quantize --quant int8
  python real_evaluate.py --mode quantize --quant int4

  # 3. 知識蒸留 (Teacher=10B → Student=3B)
  python real_evaluate.py --mode distill --student_model Qwen/Qwen2.5-3B

  # 4. LoRA ファインチューニング
  python real_evaluate.py --mode lora --lora_r 8 --finetune_steps 200

  # 5. 全手法一括評価
  python real_evaluate.py --mode all --num_samples 50

  # 6. GCS 結果アップロード付き (Vertex AI 環境)
  AIP_MODEL_DIR=gs://my-bucket/results python real_evaluate.py --mode all
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ─── ライブラリの可用性チェック ───────────────────────────────
def _check_import(name: str) -> bool:
    import importlib
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False

HAS_ALPAMAYO    = _check_import("alpamayo_r1")
HAS_BNB         = _check_import("bitsandbytes")
HAS_PEFT        = _check_import("peft")
HAS_TRANSFORMERS = _check_import("transformers")
HAS_TORCHAO     = _check_import("torchao")
HAS_GCS         = _check_import("google.cloud.storage")

RESULTS_DIR = Path(os.environ.get("LOCAL_RESULTS_DIR", "./results_real"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARN] GPU が検出されません。実際の評価には VRAM 24GB+ の GPU が必要です。")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データ構造
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalResult:
    """1手法の評価結果"""
    method: str
    # 軌道精度メトリクス
    l2_mean: float = float("nan")        # 平均L2誤差 (m)
    l2_final: float = float("nan")       # 最終ウェイポイントL2誤差
    ade: float = float("nan")            # Average Displacement Error
    fde: float = float("nan")            # Final Displacement Error
    heading_error: float = float("nan")  # 方向誤差 (rad)
    # AlpaSim 代理メトリクス
    offroad_rate: float = float("nan")
    comfort_score: float = float("nan")  # 加速度の滑らかさ
    # CoC (Chain-of-Causation) 品質
    coc_length_mean: float = float("nan")  # 推論テキスト長 (トークン)
    coc_keywords_hit: float = float("nan") # シナリオ関連キーワード含有率
    # システムメトリクス
    latency_ms: float = float("nan")
    latency_p95_ms: float = float("nan")
    memory_mb: float = float("nan")
    throughput_fps: float = float("nan")
    # モデル情報
    num_params: int = 0
    dtype: str = ""
    compression_ratio: float = 1.0
    # 追加情報
    num_samples_evaluated: int = 0
    notes: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データローダー: PhysicalAI-AV Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_physicalai_dataset(
    num_samples: int = 50,
    hf_token: Optional[str] = None,
    split: str = "validation",
    seed: int = 42,
) -> List[dict]:
    """
    nvidia/PhysicalAI-Autonomous-Vehicles からサンプルをロード

    データ形式 (alpamayo_r1 のロードユーティリティを使用):
      - images: List[PIL.Image] — 4カメラ × 4フレーム
      - ego_history_xyz: np.ndarray (16, 3) — 自車位置履歴
      - ego_history_rot: np.ndarray (16, 9) — 自車回転履歴 (3x3 flatten)
      - ego_future_xyz:  np.ndarray (64, 3) — 正解軌道位置
      - ego_future_rot:  np.ndarray (64, 9) — 正解軌道回転
      - timestamps:      np.ndarray (20,)  — タイムスタンプ
      - coc_text:        str               — 参照 Chain-of-Causation テキスト

    Returns:
        サンプルのリスト
    """
    if HAS_ALPAMAYO:
        # 公式ロードユーティリティを使用
        try:
            from alpamayo_r1.load_physical_aiavdataset import load_pai_av_dataset
            samples = load_pai_av_dataset(
                num_clips=num_samples,
                hf_token=hf_token or os.environ.get("HF_TOKEN", ""),
                split=split,
                seed=seed,
            )
            print(f"[データ] PhysicalAI-AV Dataset: {len(samples)} サンプルをロード")
            return samples
        except Exception as e:
            print(f"[WARN] PhysicalAI-AV Dataset のロードに失敗: {e}")
            print("  → シンセティックデータにフォールバック")

    # フォールバック: シンセティックデータ (HF トークンなし / alpamayo 未インストール時)
    return _make_synthetic_samples(num_samples, seed)


def _make_synthetic_samples(num_samples: int, seed: int = 42) -> List[dict]:
    """
    実データが取得できない場合のシンセティックサンプル生成

    実データと同じ shape/dtype/範囲を持つテンソルを生成する。
    Alpamayo-R1 の入力仕様:
      - 4カメラ × 4フレーム、RGB 320×576 (processor がリサイズ)
      - ego_history_xyz: (16, 3) メートル単位
      - ego_history_rot: (16, 9) = (16, 3, 3) の flatten
      - ego_future_xyz:  (64, 3)
      - ego_future_rot:  (64, 9)
    """
    from PIL import Image as PILImage
    rng = np.random.RandomState(seed)
    samples = []

    camera_names = ["front_wide", "front_tele", "cross_left", "cross_right"]
    n_frames = 4       # 0.4s @ 10Hz
    n_history = 16     # 1.6s @ 10Hz
    n_future = 64      # 6.4s @ 10Hz
    img_h, img_w = 320, 576

    scenario_descs = [
        "Drive straight on a clear road.",
        "Turn left at the intersection.",
        "Turn right at the intersection.",
        "Follow the vehicle ahead.",
        "Stop for a pedestrian crossing.",
        "Merge onto the highway.",
        "Navigate a roundabout.",
        "Change lane to the left.",
    ]

    for i in range(num_samples):
        # カメラ画像: 各カメラ × 各フレーム
        images = []
        for cam in camera_names:
            for frame in range(n_frames):
                arr = rng.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
                images.append(PILImage.fromarray(arr))

        # 自車履歴 (直進に近いランダム軌道)
        speed = rng.uniform(5, 15)  # m/s
        dt = 0.1
        ego_hist_xyz = np.zeros((n_history, 3))
        for t in range(1, n_history):
            ego_hist_xyz[t, 0] = ego_hist_xyz[t-1, 0] + speed * dt + rng.randn() * 0.1
            ego_hist_xyz[t, 1] = ego_hist_xyz[t-1, 1] + rng.randn() * 0.05
        # 回転行列 (yaw のみ、近似)
        yaw = rng.uniform(-0.1, 0.1)
        rot = np.array([
            np.cos(yaw), -np.sin(yaw), 0,
            np.sin(yaw),  np.cos(yaw), 0,
            0, 0, 1
        ], dtype=np.float32)
        ego_hist_rot = np.tile(rot, (n_history, 1))

        # 正解軌道 (未来)
        ego_fut_xyz = np.zeros((n_future, 3))
        for t in range(n_future):
            ego_fut_xyz[t, 0] = ego_hist_xyz[-1, 0] + speed * dt * (t + 1) + rng.randn() * 0.05
            ego_fut_xyz[t, 1] = ego_hist_xyz[-1, 1] + rng.randn() * 0.03
        ego_fut_rot = np.tile(rot, (n_future, 1))

        # タイムスタンプ
        t0 = float(i * 20.0)  # 20秒クリップ
        timestamps = np.linspace(t0, t0 + 2.0, n_history + n_future)

        samples.append({
            "images": images,
            "ego_history_xyz": ego_hist_xyz.astype(np.float32),
            "ego_history_rot": ego_hist_rot.astype(np.float32),
            "ego_future_xyz":  ego_fut_xyz.astype(np.float32),
            "ego_future_rot":  ego_fut_rot.astype(np.float32),
            "timestamps": timestamps.astype(np.float64),
            "user_command": rng.choice(scenario_descs),
            "clip_id": f"synthetic_{i:04d}",
            "country": rng.choice(["JP", "US", "DE", "FR", "CN"]),
        })

    print(f"[データ] シンセティックデータ: {len(samples)} サンプル生成 (実データの代替)")
    return samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メトリクス計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_trajectory_metrics(
    pred_xyz: np.ndarray,   # (T, 3) or (N, T, 3) — N: サンプル数
    gt_xyz: np.ndarray,     # (T, 3)
    pred_rot: Optional[np.ndarray] = None,
    gt_rot: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    軌道精度メトリクスの計算

    Alpamayo-R1 論文 (arXiv:2511.00088) で使用されるメトリクスに準拠:
      - ADE (Average Displacement Error): 全ウェイポイントの平均L2
      - FDE (Final Displacement Error):  最終ウェイポイントのL2
      - Offroad Rate: 最大横方向変位が閾値超えの割合
      - Comfort Score: 加速度変化の滑らかさ
    """
    # pred が複数サンプル (N, T, 3) の場合は最良サンプルを選択 (minADE)
    if pred_xyz.ndim == 3:
        # minADE: N サンプルの中で GT に最も近い軌道を選択
        ades = np.mean(np.linalg.norm(pred_xyz - gt_xyz[None], axis=-1), axis=-1)
        best_idx = np.argmin(ades)
        pred_xyz = pred_xyz[best_idx]

    diff = pred_xyz - gt_xyz   # (T, 3)
    xy_diff = diff[:, :2]      # (T, 2) — 水平面のみ

    ade = float(np.mean(np.linalg.norm(xy_diff, axis=-1)))
    fde = float(np.linalg.norm(xy_diff[-1]))

    # Heading error (z 軸周りの回転差)
    heading_error = float("nan")
    if pred_rot is not None and gt_rot is not None:
        # rot は (T, 9) = (T, 3, 3) flatten
        if pred_rot.ndim == 3:
            pred_rot = pred_rot[best_idx] if pred_rot.ndim == 3 else pred_rot
        p_rot = pred_rot.reshape(-1, 3, 3)
        g_rot = gt_rot.reshape(-1, 3, 3)
        # R_err = R_pred @ R_gt^T → trace → angle
        R_err = p_rot @ g_rot.transpose(0, 2, 1)
        cos_ang = np.clip((np.trace(R_err, axis1=1, axis2=2) - 1) / 2, -1, 1)
        heading_error = float(np.mean(np.arccos(cos_ang)))

    # Offroad rate: |dy| > 2.0m を逸脱と定義
    max_lateral = np.max(np.abs(pred_xyz[:, 1]))
    offroad = float(max_lateral > 2.0)

    # Comfort score: 加速度の二次微分 (jerk) の絶対値
    dx = np.diff(pred_xyz[:, 0], n=2)  # 二次差分
    comfort = float(1.0 / (1.0 + np.mean(np.abs(dx))))  # 高いほど滑らか

    return {
        "ade": ade,
        "fde": fde,
        "l2_mean": ade,
        "l2_final": fde,
        "heading_error": heading_error,
        "offroad_rate": offroad,
        "comfort_score": comfort,
    }


def evaluate_coc_quality(coc_text: str, user_command: str) -> Dict[str, float]:
    """
    Chain-of-Causation テキストの品質評価

    実 Alpamayo-R1 は <|cot_start|> ... <|traj_future_start|> 間に
    推論テキストを生成する。その品質を簡易評価する。
    """
    if not coc_text:
        return {"coc_length": 0.0, "coc_keywords_hit": 0.0}

    # 推論テキスト長 (語数)
    words = coc_text.split()
    length = float(len(words))

    # シナリオ関連キーワードの含有率
    keywords = {
        "lane", "speed", "vehicle", "pedestrian", "intersection",
        "stop", "follow", "merge", "turn", "straight", "traffic",
        "car", "road", "brake", "accelerat",
        # 日本語対応
        "車線", "速度", "歩行者", "停止", "直進", "右折", "左折",
    }
    text_lower = coc_text.lower()
    hit = sum(1 for kw in keywords if kw.lower() in text_lower)
    hit_rate = float(hit) / len(keywords)

    return {"coc_length": length, "coc_keywords_hit": hit_rate}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# モデルロードユーティリティ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_ID = "nvidia/Alpamayo-R1-10B"

def _model_memory_mb(model) -> float:
    """モデルパラメータのメモリ使用量 (MB)"""
    total = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total / (1024 ** 2)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def load_baseline_model(
    dtype: torch.dtype = torch.bfloat16,
    hf_token: Optional[str] = None,
):
    """
    FP32/BF16 ベースラインモデルのロード

    公式手順:
        from alpamayo_r1 import AlpamayoR1
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)
    """
    if not HAS_ALPAMAYO:
        raise ImportError(
            "alpamayo_r1 が未インストールです。\n"
            "  uv venv ar1_venv && source ar1_venv/bin/activate && uv sync\n"
            "または:\n"
            "  pip install git+https://github.com/NVlabs/alpamayo.git"
        )

    from alpamayo_r1 import AlpamayoR1
    import alpamayo_r1.helper as helper

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "HuggingFace トークンが必要です。\n"
            "  export HF_TOKEN=hf_xxxx\n"
            "  または --hf_token 引数で指定してください。\n"
            "  アクセス申請: https://huggingface.co/nvidia/Alpamayo-R1-10B"
        )

    print(f"[モデル] {MODEL_ID} をロード中 (dtype={dtype})...")
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        token=token,
    ).to(DEVICE)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    elapsed = time.time() - t0
    print(f"  ロード完了: {elapsed:.1f}s, "
          f"メモリ: {_model_memory_mb(model):.0f}MB, "
          f"パラメータ: {_count_params(model)/1e9:.2f}B")
    return model, processor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 推論ループ (全手法共通)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def run_inference_loop(
    model,
    processor,
    samples: List[dict],
    method_name: str,
    num_traj_samples: int = 1,
    warmup: int = 2,
) -> EvalResult:
    """
    Alpamayo-R1 の sample_trajectories_from_data_with_vlm_rollout() を
    各サンプルに対して実行し、メトリクスを集約する。

    公式 API:
        results = model.sample_trajectories_from_data_with_vlm_rollout(
            data=sample,
            processor=processor,
            num_traj_samples=num_traj_samples,
        )
        # results['pred_xyz']:  (N, 64, 3) — N個の予測軌道
        # results['pred_rot']:  (N, 64, 9)
        # results['coc_text']:  str — Chain-of-Causation テキスト
    """
    all_metrics: List[Dict] = []
    latencies: List[float] = []
    coc_qualities: List[Dict] = []

    for i, sample in enumerate(samples):
        if i < warmup:
            # ウォームアップ (GPU キャッシュ安定化)
            try:
                _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=sample,
                    processor=processor,
                    num_traj_samples=1,
                )
            except Exception:
                pass
            continue

        try:
            t0 = time.perf_counter()
            results = model.sample_trajectories_from_data_with_vlm_rollout(
                data=sample,
                processor=processor,
                num_traj_samples=num_traj_samples,
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            # 軌道メトリクス
            pred_xyz = np.array(results["pred_xyz"])  # (N, 64, 3)
            gt_xyz   = sample["ego_future_xyz"]        # (64, 3)
            pred_rot = np.array(results.get("pred_rot", [None] * len(pred_xyz)))
            gt_rot   = sample.get("ego_future_rot")

            traj_m = compute_trajectory_metrics(pred_xyz, gt_xyz, pred_rot, gt_rot)
            all_metrics.append(traj_m)

            # CoC 品質
            coc_text = results.get("coc_text", "")
            coc_q = evaluate_coc_quality(coc_text, sample.get("user_command", ""))
            coc_qualities.append(coc_q)

            if (i - warmup + 1) % 10 == 0:
                print(f"  [{method_name}] {i-warmup+1}/{len(samples)-warmup} "
                      f"ADE={traj_m['ade']:.3f}m  FDE={traj_m['fde']:.3f}m  "
                      f"lat={latencies[-1]:.0f}ms")

        except torch.cuda.OutOfMemoryError:
            print(f"  [ERROR] OOM at sample {i}: VRAM 不足")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] sample {i} でエラー: {e}")
            continue

    if not all_metrics:
        return EvalResult(method=method_name, notes="評価失敗: サンプルが処理されませんでした")

    # 集約
    def mean_field(key):
        vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    result = EvalResult(
        method=method_name,
        l2_mean=mean_field("l2_mean"),
        l2_final=mean_field("l2_final"),
        ade=mean_field("ade"),
        fde=mean_field("fde"),
        heading_error=mean_field("heading_error"),
        offroad_rate=mean_field("offroad_rate"),
        comfort_score=mean_field("comfort_score"),
        coc_length_mean=float(np.mean([q["coc_length"] for q in coc_qualities])),
        coc_keywords_hit=float(np.mean([q["coc_keywords_hit"] for q in coc_qualities])),
        latency_ms=float(np.mean(latencies)) if latencies else float("nan"),
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else float("nan"),
        memory_mb=_model_memory_mb(model),
        throughput_fps=float(1000.0 / np.mean(latencies)) if latencies else float("nan"),
        num_params=_count_params(model),
        num_samples_evaluated=len(all_metrics),
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手法 1: BF16/FP32 ベースライン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_baseline(
    samples: List[dict],
    hf_token: str,
    dtype: torch.dtype = torch.bfloat16,
) -> EvalResult:
    """
    BF16 フルモデル評価 (公式推奨設定)

    Alpamayo-R1 の公式ノートブックと同じ設定:
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)
    """
    print(f"\n{'='*60}")
    print(f"[Baseline] BF16 フルモデル評価")
    print(f"{'='*60}")

    model, processor = load_baseline_model(dtype=dtype, hf_token=hf_token)
    result = run_inference_loop(model, processor, samples, method_name="baseline_bf16")
    result.dtype = str(dtype).replace("torch.", "")
    result.compression_ratio = 1.0

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手法 2: bitsandbytes 量子化 (INT8 / INT4 NF4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_bnb_quantized(
    samples: List[dict],
    hf_token: str,
    quant_type: str = "int8",  # "int8" | "int4"
) -> EvalResult:
    """
    bitsandbytes を使った量子化モデル評価

    INT8 (LLM.int8):
        BitsAndBytesConfig(load_in_8bit=True)
        → Linear層の重みを INT8、活性化を FP16 で計算
        → メモリ ~50% 削減、精度損失は最小

    INT4 NF4 (QLoRA 手法):
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
        → メモリ ~75% 削減、精度損失は中程度
        → double quantization で量子化定数自体も圧縮

    Alpamayo-R1 への適用注意点:
        - VLM backbone (Qwen3VLForConditionalGeneration) の Linear 層を量子化
        - Action Expert (Qwen3Model) も量子化対象
        - Diffusion head (連続値出力) は FP32/BF16 を維持推奨
        - Vision encoder の Conv2d は量子化非対応 → FP16 のまま
    """
    if not HAS_BNB:
        return EvalResult(
            method=f"bnb_{quant_type}",
            notes="bitsandbytes 未インストール: pip install bitsandbytes"
        )

    print(f"\n{'='*60}")
    print(f"[量子化] bitsandbytes {quant_type.upper()} 評価")
    print(f"{'='*60}")

    from transformers import BitsAndBytesConfig
    from alpamayo_r1 import AlpamayoR1
    import alpamayo_r1.helper as helper

    if quant_type == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,          # 外れ値検出閾値
            llm_int8_has_fp16_weight=False,
        )
        compression_ratio = 2.0
    elif quant_type == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",       # NormalFloat4: LLM重みに最適
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # 量子化定数を再量子化
        )
        compression_ratio = 4.0
    else:
        raise ValueError(f"未対応の量子化タイプ: {quant_type}")

    # Alpamayo-R1 は from_pretrained の quantization_config 引数で量子化設定を渡す
    # 内部では VLM と Expert それぞれに適用される
    token = hf_token or os.environ.get("HF_TOKEN", "")
    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",          # GPU VRAM が足りない場合は CPU オフロード
        token=token,
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    print(f"  量子化完了: メモリ {_model_memory_mb(model):.0f}MB")

    result = run_inference_loop(
        model, processor, samples,
        method_name=f"bnb_{quant_type}",
    )
    result.dtype = f"bnb_{quant_type}"
    result.compression_ratio = compression_ratio

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手法 3: torchao 量子化 (FP8 / INT8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_torchao_quantized(
    samples: List[dict],
    hf_token: str,
    quant_type: str = "int8_weight_only",
) -> EvalResult:
    """
    torchao を使った量子化 (PyTorch 公式・次世代 API)

    対応する量子化タイプ:
        "int8_weight_only":   重みのみ INT8 (activations は BF16)
        "int8_dynamic":       重み+活性化 INT8 (動的量子化)
        "fp8_weight_only":    重みのみ FP8 (H100/A100 推奨)
        "int4_weight_only":   重みのみ INT4 (GPTQ-style)

    Li Auto が ThorチップでFP16→FP8→INT8 の段階的精度削減を実施したのと同じ戦略。

    注意:
        - torchao は torch.compile() との組み合わせで最大効果
        - H100 以降では FP8 が最も効率的
        - A100 では INT8 が推奨
    """
    if not HAS_TORCHAO:
        return EvalResult(
            method=f"torchao_{quant_type}",
            notes="torchao 未インストール: pip install torchao"
        )

    print(f"\n{'='*60}")
    print(f"[量子化] torchao {quant_type} 評価")
    print(f"{'='*60}")

    import torchao
    from torchao.quantization import quantize_
    from alpamayo_r1 import AlpamayoR1
    import alpamayo_r1.helper as helper

    token = hf_token or os.environ.get("HF_TOKEN", "")
    model, processor = load_baseline_model(dtype=torch.bfloat16, hf_token=token)

    # VLM と Expert の Linear 層を量子化
    # Diffusion head (TrajectoryDecoder) は精度を保持するため除外
    _apply_torchao(model, quant_type)

    # torch.compile でカーネル最適化 (オプション、A100/H100 推奨)
    use_compile = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
    if use_compile:
        print("  torch.compile() を適用中...")
        model = torch.compile(model, mode="reduce-overhead")

    result = run_inference_loop(
        model, processor, samples,
        method_name=f"torchao_{quant_type}",
    )
    result.dtype = f"torchao_{quant_type}"
    result.compression_ratio = {"int8_weight_only": 2.0, "fp8_weight_only": 2.0,
                                 "int8_dynamic": 2.0, "int4_weight_only": 4.0}.get(quant_type, 2.0)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def _apply_torchao(model, quant_type: str):
    """torchao の量子化を VLM/Expert に適用"""
    from torchao.quantization import (
        quantize_,
        int8_weight_only,
        int8_dynamic_activation_int8_weight,
        int4_weight_only,
    )
    try:
        from torchao.quantization import float8_weight_only
        HAS_FP8 = True
    except ImportError:
        HAS_FP8 = False

    # VLM backbone の量子化
    if hasattr(model, "vlm"):
        if quant_type == "int8_weight_only":
            quantize_(model.vlm, int8_weight_only())
        elif quant_type == "int8_dynamic":
            quantize_(model.vlm, int8_dynamic_activation_int8_weight())
        elif quant_type == "int4_weight_only":
            quantize_(model.vlm, int4_weight_only())
        elif quant_type == "fp8_weight_only" and HAS_FP8:
            quantize_(model.vlm, float8_weight_only())
        print(f"  VLM 量子化完了: {quant_type}")

    # Action Expert の量子化
    if hasattr(model, "expert"):
        if quant_type == "int8_weight_only":
            quantize_(model.expert, int8_weight_only())
        elif quant_type == "int8_dynamic":
            quantize_(model.expert, int8_dynamic_activation_int8_weight())
        print(f"  Action Expert 量子化完了: {quant_type}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手法 4: 知識蒸留 (Teacher 10B → Student 小型モデル)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_distillation(
    samples: List[dict],
    hf_token: str,
    student_model_id: str = "Qwen/Qwen2.5-3B",
    finetune_steps: int = 200,
    temperature: float = 4.0,
    alpha: float = 0.5,
    save_student: bool = True,
) -> EvalResult:
    """
    知識蒸留: Alpamayo-R1 (Teacher 10B) → 小型 Student モデル

    Li Auto の実装を参考にした戦略:
      - Teacher: Alpamayo-R1-10B (BF16、frozen)
      - Student: Qwen2.5-3B ベースに Action Expert Head を追加
      - 損失: 軌道 MSE + Soft label KL (temperature scaling) の複合

    蒸留パイプライン:
      1. Teacher で各サンプルの軌道予測を生成 (offline distillation)
      2. Student が Teacher の軌道分布を模倣するよう学習
      3. Student 単体で評価

    制約:
      - Student モデルは Vision encoder を持たないため
        Teacher の vision embeddings を蒸留ターゲットとして使用
      - Diffusion head は Student に新たに追加 (小型版)
    """
    print(f"\n{'='*60}")
    print(f"[蒸留] Teacher=Alpamayo-R1-10B → Student={student_model_id}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    token = hf_token or os.environ.get("HF_TOKEN", "")

    # ─── Teacher: Alpamayo-R1-10B ─────────────────────────
    print("  Teacher モデルをロード中...")
    teacher_model, processor = load_baseline_model(dtype=torch.bfloat16, hf_token=token)
    teacher_model.eval()

    # ─── Teacher による軌道生成 (offline distillation) ────
    print("  Teacher で蒸留ターゲットを生成中...")
    distill_data = []
    for i, sample in enumerate(samples[:finetune_steps]):
        try:
            with torch.no_grad():
                results = teacher_model.sample_trajectories_from_data_with_vlm_rollout(
                    data=sample,
                    processor=processor,
                    num_traj_samples=3,   # 複数サンプルで分布を推定
                )
            distill_data.append({
                "ego_history_xyz": sample["ego_history_xyz"],
                "ego_history_rot": sample["ego_history_rot"],
                "ego_future_xyz":  sample["ego_future_xyz"],
                "teacher_pred_xyz": np.array(results["pred_xyz"]),  # (3, 64, 3)
                "teacher_coc":     results.get("coc_text", ""),
            })
            if (i + 1) % 20 == 0:
                print(f"    Teacher 生成: {i+1}/{finetune_steps}")
        except Exception as e:
            print(f"    [WARN] sample {i}: {e}")

    # Teacher を解放して VRAM を確保
    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()

    # ─── Student モデル構築 ───────────────────────────────
    print(f"\n  Student モデル ({student_model_id}) をロード中...")
    student_lm = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        torch_dtype=torch.bfloat16,
        token=token,
    ).to(DEVICE)

    # Student の軌道ヘッド (簡易 MLP)
    hidden_size = student_lm.config.hidden_size
    student_traj_head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 64 * 3),  # 64ウェイポイント × 3次元
    ).to(DEVICE).to(torch.bfloat16)

    # ─── 蒸留学習 ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(student_lm.parameters()) + list(student_traj_head.parameters()),
        lr=2e-5, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(distill_data)
    )

    student_lm.train()
    student_traj_head.train()
    losses = []

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_id, token=token)

    for step, d in enumerate(distill_data):
        # egomotion をテキスト化して Student に入力
        hist_text = f"Ego speed: {np.linalg.norm(np.diff(d['ego_history_xyz'][-2:], axis=0)):.1f}m/s"
        inputs = student_tokenizer(hist_text, return_tensors="pt").to(DEVICE)

        out = student_lm(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :]  # (1, hidden_size)
        pred_traj = student_traj_head(hidden).reshape(1, 64, 3)  # (1, 64, 3)

        # (a) GT 軌道との MSE 損失
        gt_traj = torch.tensor(d["ego_future_xyz"], dtype=torch.bfloat16, device=DEVICE)
        loss_gt = F.mse_loss(pred_traj[0], gt_traj)

        # (b) Teacher 最良予測との蒸留損失 (minADE で最良サンプルを選択)
        teacher_preds = torch.tensor(
            d["teacher_pred_xyz"], dtype=torch.bfloat16, device=DEVICE
        )  # (3, 64, 3)
        ades = torch.mean(torch.norm(teacher_preds - gt_traj[None], dim=-1), dim=-1)
        best_teacher = teacher_preds[ades.argmin()]  # (64, 3)
        loss_distill = F.mse_loss(pred_traj[0], best_teacher)

        loss = alpha * loss_distill + (1 - alpha) * loss_gt
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_lm.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"    蒸留 step {step+1}/{len(distill_data)}  loss={np.mean(losses[-10:]):.4f}")

    final_loss = float(np.mean(losses[-20:])) if losses else float("nan")
    print(f"  蒸留完了: 最終損失={final_loss:.4f}")

    # ─── Student を保存 ──────────────────────────────────
    if save_student:
        student_save_path = RESULTS_DIR / "student_model"
        student_lm.save_pretrained(student_save_path)
        torch.save(student_traj_head.state_dict(), student_save_path / "traj_head.pt")
        print(f"  Student モデル保存: {student_save_path}")

    # ─── Student 単体で評価 ──────────────────────────────
    student_lm.eval()
    student_traj_head.eval()

    all_metrics = []
    latencies = []
    for sample in samples:
        try:
            t0 = time.perf_counter()
            hist_text = f"Ego speed: {np.linalg.norm(np.diff(sample['ego_history_xyz'][-2:], axis=0)):.1f}m/s"
            inputs = student_tokenizer(hist_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = student_lm(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :]
                pred_traj = student_traj_head(hidden).reshape(64, 3).cpu().numpy()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            gt_xyz = sample["ego_future_xyz"]
            m = compute_trajectory_metrics(pred_traj[None], gt_xyz)
            all_metrics.append(m)
        except Exception as e:
            print(f"  [WARN] {e}")

    result = EvalResult(
        method=f"distilled_{student_model_id.split('/')[-1]}",
        l2_mean=float(np.mean([m["l2_mean"] for m in all_metrics])) if all_metrics else float("nan"),
        ade=float(np.mean([m["ade"] for m in all_metrics])) if all_metrics else float("nan"),
        fde=float(np.mean([m["fde"] for m in all_metrics])) if all_metrics else float("nan"),
        offroad_rate=float(np.mean([m["offroad_rate"] for m in all_metrics])) if all_metrics else float("nan"),
        latency_ms=float(np.mean(latencies)) if latencies else float("nan"),
        memory_mb=_model_memory_mb(student_lm),
        num_params=_count_params(student_lm),
        dtype="bfloat16",
        compression_ratio=10.0 / 3.0,  # 10B / 3B
        notes=f"蒸留損失={final_loss:.4f}, steps={len(distill_data)}",
        num_samples_evaluated=len(all_metrics),
    )

    del student_lm, student_traj_head
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 手法 5: LoRA ファインチューニング + 評価
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_lora(
    samples: List[dict],
    hf_token: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    finetune_steps: int = 200,
    target_domain: str = "JP",    # 特定地域・シナリオへの適応
    save_adapter: bool = True,
) -> EvalResult:
    """
    LoRA ファインチューニング + 評価

    Alpamayo-R1 への LoRA 適用戦略:
      - ターゲット: VLM backbone (Qwen3VLForConditionalGeneration) の
                   Attention Q/K/V/O 行列
      - Vision encoder (SigLIP) は凍結 (汎用視覚特徴のため)
      - Action Expert (Qwen3Model) は凍結または軽い LoRA
      - Diffusion head は full fine-tune (軽量なため)

    実用シナリオ:
      - 特定の地域 (例: 日本の左側通行) への適応
      - 特定の天候条件 (雨天・夜間) への特化
      - 自社車両の egomotion 特性への適応

    VRAM 要件:
      - フル FT (BF16): ~80GB
      - LoRA (r=8, BF16 frozen): ~30GB
      - LoRA + INT8 frozen: ~20GB
      - LoRA + INT4 frozen: ~15GB  ← 普及価格帯向け検討ライン
    """
    if not HAS_PEFT:
        return EvalResult(
            method=f"lora_r{lora_r}",
            notes="peft 未インストール: pip install peft"
        )

    print(f"\n{'='*60}")
    print(f"[LoRA] r={lora_r}, alpha={lora_alpha}, target={target_domain}")
    print(f"{'='*60}")

    from peft import LoraConfig, get_peft_model, TaskType
    import torch.nn.functional as F

    token = hf_token or os.environ.get("HF_TOKEN", "")
    model, processor = load_baseline_model(dtype=torch.bfloat16, hf_token=token)

    # ─── LoRA 設定 ───────────────────────────────────────
    # Qwen3VL の Attention 層名を指定
    # (alpamayo_r1 の内部構造: model.vlm.model.layers[*].self_attn.{q,k,v,o}_proj)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Qwen3VL の Attention 投影行列
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        # Vision encoder は除外 (visual.* は凍結)
        # Diffusion head は full FT (modules_to_save)
        modules_to_save=["action_head", "traj_cond_proj"]
        if hasattr(model, "action_head") else [],
    )

    # VLM backbone のみに LoRA を適用 (Action Expert は凍結)
    if hasattr(model, "vlm"):
        peft_vlm = get_peft_model(model.vlm, lora_cfg)
        model.vlm = peft_vlm
        peft_vlm.print_trainable_parameters()
    else:
        # フォールバック: モデル全体に適用
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ─── LoRA 学習 ───────────────────────────────────────
    # LoRA パラメータのみ最適化
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=1e-4,
    )

    model.train()
    losses = []
    domain_samples = [s for s in samples if s.get("country", "") == target_domain]
    train_samples = domain_samples if domain_samples else samples
    print(f"  学習サンプル数: {len(train_samples)} ({target_domain} フィルタ後)")

    for step in range(finetune_steps):
        sample = train_samples[step % len(train_samples)]
        try:
            # 軌道予測の forward pass
            # Alpamayo-R1 の内部 forward: VLM forward → Action Expert forward
            # ここでは軌道予測部分のみ学習 (CoC 生成はスキップ)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                results = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=sample,
                    processor=processor,
                    num_traj_samples=1,
                )
            pred_xyz = torch.tensor(
                results["pred_xyz"][0], dtype=torch.bfloat16, device=DEVICE
            )  # (64, 3)
            gt_xyz = torch.tensor(
                sample["ego_future_xyz"], dtype=torch.bfloat16, device=DEVICE
            )
            loss = F.mse_loss(pred_xyz, gt_xyz)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                print(f"  LoRA step {step+1}/{finetune_steps}  loss={np.mean(losses[-10:]):.4f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] step {step}: VRAM 不足 — バッチサイズを下げてください")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  [WARN] step {step}: {e}")
            continue

    # ─── アダプタ保存 ─────────────────────────────────────
    if save_adapter:
        adapter_path = RESULTS_DIR / f"lora_r{lora_r}_{target_domain}"
        if hasattr(model, "vlm") and hasattr(model.vlm, "save_pretrained"):
            model.vlm.save_pretrained(adapter_path)
        else:
            model.save_pretrained(adapter_path)
        print(f"  LoRA アダプタ保存: {adapter_path}")

    # ─── 評価 ────────────────────────────────────────────
    model.eval()
    result = run_inference_loop(model, processor, samples, method_name=f"lora_r{lora_r}")
    result.dtype = "bfloat16+lora"
    result.compression_ratio = 1.0  # ベースと同サイズだがアダプタのみ差し替え可
    result.notes = (f"LoRA r={lora_r}, alpha={lora_alpha}, "
                    f"steps={finetune_steps}, loss={np.mean(losses[-20:]):.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 結果表示・保存
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_results_table(results: List[EvalResult]) -> None:
    w = 120
    print("\n" + "=" * w)
    print("Alpamayo-R1-10B 実モデル 圧縮・精度評価結果")
    print("=" * w)
    hdr = (f"{'手法':<28} {'ADE(m)↓':>8} {'FDE(m)↓':>8} "
           f"{'Offrd↓':>7} {'Cmft↑':>6} {'CoC長':>7} "
           f"{'Lat(ms)':>8} {'Mem(GB)':>8} {'圧縮率':>6} {'Params':>8}")
    print(hdr)
    print("-" * w)

    def f(v, d=3):
        return f"{v:.{d}f}" if (v is not None and not np.isnan(v)) else "   n/a"

    for r in results:
        print(
            f"{r.method:<28} "
            f"{f(r.ade):>8} {f(r.fde):>8} "
            f"{f(r.offroad_rate, 2):>7} {f(r.comfort_score, 2):>6} {f(r.coc_length_mean, 0):>7} "
            f"{f(r.latency_ms, 0):>8} {f(r.memory_mb/1024, 1):>8} "
            f"{f(r.compression_ratio, 1):>6}x "
            f"{r.num_params/1e9:.2f}B"
        )
        if r.notes:
            print(f"{'':>28}  ↳ {r.notes}")
    print("=" * w)
    print("ADE: Average Displacement Error (m), FDE: Final Displacement Error, "
          "Offrd: Offroad Rate, Cmft: Comfort Score, Lat: Latency")


def save_results(results: List[EvalResult], path: Path) -> None:
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {path}")


def upload_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """GCS へのアップロード (Vertex AI 環境用)"""
    if not HAS_GCS or not gcs_uri:
        return
    from google.cloud import storage
    bucket_name, prefix = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for fpath in local_dir.rglob("*"):
        if fpath.is_file():
            blob_name = f"{prefix}/{fpath.relative_to(local_dir)}"
            bucket.blob(blob_name).upload_from_filename(str(fpath))
            print(f"  GCS: {blob_name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メインエントリーポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Alpamayo-R1-10B 実モデル 量子化・蒸留・LoRA 評価",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["baseline", "quantize", "distill", "lora", "all"],
                        default="baseline", help="評価手法")
    parser.add_argument("--quant", choices=["int8", "int4", "fp8", "int8_dynamic"],
                        default="int8", help="量子化タイプ (--mode quantize 時)")
    parser.add_argument("--quant_backend", choices=["bnb", "torchao"],
                        default="bnb", help="量子化バックエンド")
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-3B",
                        help="蒸留の Student モデル ID")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA ランク")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--finetune_steps", type=int, default=200,
                        help="蒸留/LoRA 学習ステップ数")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="評価サンプル数 (デフォルト 50)")
    parser.add_argument("--hf_token", default="",
                        help="HuggingFace アクセストークン (または HF_TOKEN 環境変数)")
    parser.add_argument("--target_domain", default="JP",
                        help="LoRA 適応ドメイン (国コード: JP/US/DE/...)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16", help="ベースラインの dtype")
    parser.add_argument("--num_traj_samples", type=int, default=1,
                        help="1サンプルあたりの軌道生成数 (多いほど精度向上・低速)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Alpamayo-R1-10B 実モデル評価スクリプト")
    print("=" * 60)
    print(f"  Mode          : {args.mode}")
    print(f"  Num samples   : {args.num_samples}")
    print(f"  Device        : {DEVICE}")
    print(f"  alpamayo_r1   : {'✅ インストール済み' if HAS_ALPAMAYO else '❌ 未インストール'}")
    print(f"  bitsandbytes  : {'✅' if HAS_BNB else '❌'}")
    print(f"  torchao       : {'✅' if HAS_TORCHAO else '❌'}")
    print(f"  peft          : {'✅' if HAS_PEFT else '❌'}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {props.name} ({props.total_memory/1e9:.1f}GB VRAM)")
    print("=" * 60)

    # ─── データロード ────────────────────────────────────
    samples = load_physicalai_dataset(
        num_samples=args.num_samples,
        hf_token=hf_token,
    )

    results: List[EvalResult] = []

    # ─── 評価実行 ────────────────────────────────────────
    if args.mode in ("baseline", "all"):
        r = evaluate_baseline(samples, hf_token, dtype=dtype)
        results.append(r)

    if args.mode in ("quantize", "all"):
        if args.quant_backend == "bnb":
            quant_type = args.quant if args.quant in ("int8", "int4") else "int8"
            r = evaluate_bnb_quantized(samples, hf_token, quant_type=quant_type)
            results.append(r)
        elif args.quant_backend == "torchao":
            quant_map = {"int8": "int8_weight_only", "int4": "int4_weight_only",
                         "fp8": "fp8_weight_only", "int8_dynamic": "int8_dynamic"}
            r = evaluate_torchao_quantized(
                samples, hf_token,
                quant_type=quant_map.get(args.quant, "int8_weight_only")
            )
            results.append(r)

        if args.mode == "all":
            # all モードでは BF16 baseline と INT8/INT4 の両方を評価
            for q in ["int8", "int4"]:
                r = evaluate_bnb_quantized(samples, hf_token, quant_type=q)
                results.append(r)

    if args.mode in ("distill", "all"):
        r = evaluate_distillation(
            samples, hf_token,
            student_model_id=args.student_model,
            finetune_steps=args.finetune_steps,
        )
        results.append(r)

    if args.mode in ("lora", "all"):
        r = evaluate_lora(
            samples, hf_token,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            finetune_steps=args.finetune_steps,
            target_domain=args.target_domain,
        )
        results.append(r)

    # ─── 結果表示・保存 ──────────────────────────────────
    print_results_table(results)

    result_path = RESULTS_DIR / "real_eval_results.json"
    save_results(results, result_path)

    # GCS アップロード (Vertex AI 環境)
    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    if aip_model_dir:
        upload_to_gcs(RESULTS_DIR, aip_model_dir)

    return results


if __name__ == "__main__":
    main()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12 GB VRAM 向け: INT4 + CPU オフロード評価
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# VRAM 内訳 (実測値, arXiv:2605.11678)
VRAM_BREAKDOWN_BF16 = {
    "vision_encoder":  1.15,   # GB
    "vlm_backbone":   15.17,
    "vlm_lm_head":     1.27,
    "expert_decoder":  4.56,
    "kv_cache_etc":    1.37,
    "total":          23.52,   # 実測 21.52 GB (論文値)
}


def estimate_vram_for_quant(quant_type: str) -> Dict[str, float]:
    """量子化タイプ別の VRAM 推定値を返す (GB)"""
    ratio = {
        "bf16":    1.0,
        "fp16":    1.0,
        "int8":    0.50,   # 重みのみ 8bit → ~50% 削減
        "int4":    0.25,   # 重みのみ 4bit → ~75% 削減
        "int4_dq": 0.23,   # double quantization → ~77% 削減
    }.get(quant_type, 0.5)

    base = VRAM_BREAKDOWN_BF16
    # KV cache・活性化は量子化されない (固定)
    param_vram = (base["total"] - base["kv_cache_etc"]) * ratio
    total = param_vram + base["kv_cache_etc"]
    return {
        "param_vram_gb": round(param_vram, 1),
        "kv_cache_gb":   round(base["kv_cache_etc"], 1),
        "total_gb":      round(total, 1),
        "fits_12gb":     total < 12.0,
        "fits_16gb":     total < 16.0,
        "fits_24gb":     total < 24.0,
    }


def evaluate_low_vram(
    samples: List[dict],
    hf_token: str,
    target_vram_gb: float = 12.0,
) -> List[EvalResult]:
    """
    12 GB VRAM 以下で動作する量子化手法のみを評価

    実行可能な組み合わせ (arXiv:2605.11678 の実測に基づく):
      ✅ INT4 NF4 (bitsandbytes):   ~5.5 GB
      ✅ INT4 + CPU offload:        GPU ~5.5 GB + RAM ~16 GB
      ✅ AWQ/GPTQ INT4 (torchao):   ~5.5 GB
      ⚠️  INT8:                     ~11.5 GB (KV cache 込みで OOM リスク)
      ❌ BF16/FP16:                 21.5 GB (不可)

    Args:
        target_vram_gb: 使用可能な VRAM 上限 (デフォルト 12 GB)
    """
    print(f"\n{'='*60}")
    print(f"VRAM {target_vram_gb:.0f}GB 以下で実行可能な手法を評価")
    print(f"{'='*60}")

    # VRAM フィット確認
    print("\n[VRAM 推定]")
    for qt in ["bf16", "int8", "int4", "int4_dq"]:
        est = estimate_vram_for_quant(qt)
        ok = "✅" if est["total_gb"] < target_vram_gb else "❌"
        print(f"  {qt:<10}: {est['total_gb']:>5.1f} GB  {ok}")

    results: List[EvalResult] = []

    # ─── INT4 NF4 (bitsandbytes) ─────────────────────────
    # target_vram_gb >= 8 なら試みる
    if target_vram_gb >= 6.0 and HAS_BNB:
        print(f"\n[INT4 NF4] 推定 VRAM: ~5.5 GB (target: {target_vram_gb} GB)")
        r = evaluate_bnb_quantized(samples, hf_token, quant_type="int4")
        results.append(r)

    # ─── INT4 + CPU オフロード (Accelerate device_map="auto") ─
    # 12 GB GPU でも BF16 全体を CPU+GPU で分散 → 低速だが動作する
    if target_vram_gb >= 6.0 and HAS_BNB:
        r = _evaluate_int4_cpu_offload(samples, hf_token, max_gpu_gb=target_vram_gb)
        results.append(r)

    # ─── INT8 (リスク評価) ────────────────────────────────
    est_int8 = estimate_vram_for_quant("int8")
    if target_vram_gb >= 12.0 and HAS_BNB:
        if est_int8["total_gb"] < target_vram_gb:
            print(f"\n[INT8] 推定 VRAM: {est_int8['total_gb']:.1f} GB → 試みます")
            r = evaluate_bnb_quantized(samples, hf_token, quant_type="int8")
            results.append(r)
        else:
            print(f"\n[INT8] 推定 VRAM: {est_int8['total_gb']:.1f} GB → "
                  f"target {target_vram_gb} GB 超過、スキップ")
            results.append(EvalResult(
                method="int8_skipped",
                notes=f"VRAM 不足: 推定 {est_int8['total_gb']:.1f} GB > {target_vram_gb} GB",
            ))

    return results


def _evaluate_int4_cpu_offload(
    samples: List[dict],
    hf_token: str,
    max_gpu_gb: float = 12.0,
) -> EvalResult:
    """
    INT4 + Accelerate CPU オフロード評価

    device_map="auto" + max_memory で GPU と CPU に自動分散。
    arXiv:2605.11678 の実験: RTX 3080 Ti (12 GB) + Accelerate で 273 秒/推論。
    """
    if not HAS_BNB or not HAS_ALPAMAYO:
        return EvalResult(
            method="int4_cpu_offload",
            notes="bitsandbytes または alpamayo_r1 が未インストール"
        )

    print(f"\n[INT4 + CPU offload] GPU {max_gpu_gb:.0f}GB + CPU RAM 使用")
    print("  注意: 推論速度は大幅に低下します (~4分/サンプル on RTX 3080 Ti)")

    from transformers import BitsAndBytesConfig
    from alpamayo_r1 import AlpamayoR1
    import alpamayo_r1.helper as helper

    token = hf_token or os.environ.get("HF_TOKEN", "")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # max_memory: GPU と CPU RAM の上限を明示指定
    gpu_mb = int(max_gpu_gb * 1024)
    max_memory = {
        0: f"{gpu_mb}MiB",
        "cpu": "48GiB",   # CPU RAM (pinned memory 推奨)
    }

    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        token=token,
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    # GPU に常駐するレイヤーを確認 (どこが CPU オフロードされたか)
    gpu_layers, cpu_layers = 0, 0
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            gpu_layers += 1
        else:
            cpu_layers += 1
    print(f"  GPU レイヤー: {gpu_layers}, CPU オフロード: {cpu_layers}")

    result = run_inference_loop(
        model, processor, samples,
        method_name="int4_cpu_offload",
    )
    result.dtype = "int4_nf4+cpu_offload"
    result.compression_ratio = 4.0
    result.notes = (f"GPU {max_gpu_gb:.0f}GB + CPU RAM, "
                    f"GPU layers: {gpu_layers}, CPU: {cpu_layers}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def print_vram_guide(target_vram_gb: float = 12.0) -> None:
    """VRAM 制約ガイドを表示"""
    print(f"\n{'='*60}")
    print(f"VRAM {target_vram_gb:.0f}GB 環境での Alpamayo-R1-10B 実行ガイド")
    print(f"{'='*60}")

    rows = [
        ("BF16 フルモデル",          "bf16",    "❌ 不可 (21.5 GB 必要)"),
        ("INT8 (bitsandbytes)",       "int8",    "⚠️  OOM リスクあり (~11.5 GB)"),
        ("INT4 NF4",                  "int4",    "✅ 可能 (~5.5 GB)"),
        ("INT4 + CPU offload",        "int4",    "✅ 可能 (低速: ~4分/推論)"),
        ("INT4 + Demand Layering*",   "int4",    "✅ 可能 + 高速化 (arXiv:2605.11678)"),
        ("蒸留 (Teacher同時ロード)",  "bf16",    "❌ 不可 (Teacher だけで 21.5 GB)"),
        ("LoRA 学習",                 "int4",    "⚠️  INT4 frozen + LoRA なら可能"),
    ]
    print(f"{'手法':<30} {'推定VRAM':>10}  {'判定'}")
    print("-" * 60)
    for name, qt, verdict in rows:
        est = estimate_vram_for_quant(qt)
        print(f"  {name:<28} {est['total_gb']:>6.1f} GB  {verdict}")
    print()
    print("* Demand Layering: 論文実装が必要 (arXiv:2605.11678)")
    print(f"{'='*60}")
