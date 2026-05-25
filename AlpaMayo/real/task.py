"""
trainer/task.py
Alpamayo-R1-10B 実モデル評価 — Vertex AI エントリーポイント

環境変数:
  AIP_MODEL_DIR    : gs://bucket/path  (Vertex AI が自動設定)
  HF_TOKEN         : HuggingFace アクセストークン (必須)
  EVAL_MODE        : baseline | quantize | distill | lora | all
  EVAL_QUANT       : int8 | int4 | fp8 (量子化タイプ)
  EVAL_QUANT_BACKEND: bnb | torchao
  EVAL_SAMPLES     : 評価サンプル数 (デフォルト 50)
  EVAL_FINETUNE_STEPS: 蒸留/LoRA ステップ数 (デフォルト 200)
  EVAL_STUDENT_MODEL: 蒸留 Student モデル ID
  EVAL_LORA_R      : LoRA ランク (デフォルト 8)
  EVAL_TARGET_DOMAIN: LoRA 適応ドメイン (デフォルト JP)
"""

from __future__ import annotations
import os, sys, json, traceback
from pathlib import Path


def main():
    app_dir = Path(__file__).parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    aip_model_dir  = os.environ.get("AIP_MODEL_DIR", "")
    hf_token       = os.environ.get("HF_TOKEN", "")
    mode           = os.environ.get("EVAL_MODE",          "baseline")
    quant          = os.environ.get("EVAL_QUANT",         "int8")
    quant_backend  = os.environ.get("EVAL_QUANT_BACKEND", "bnb")
    num_samples    = int(os.environ.get("EVAL_SAMPLES",   "50"))
    finetune_steps = int(os.environ.get("EVAL_FINETUNE_STEPS", "200"))
    student_model  = os.environ.get("EVAL_STUDENT_MODEL", "Qwen/Qwen2.5-3B")
    lora_r         = int(os.environ.get("EVAL_LORA_R",   "8"))
    target_domain  = os.environ.get("EVAL_TARGET_DOMAIN", "JP")

    print("=" * 60)
    print("Alpamayo-R1-10B 実モデル評価 — Vertex AI ジョブ")
    print("=" * 60)
    print(f"  AIP_MODEL_DIR : {aip_model_dir or '(ローカル)'}")
    print(f"  HF_TOKEN      : {'設定済み' if hf_token else '未設定 ⚠️'}")
    print(f"  EVAL_MODE     : {mode}")
    print(f"  EVAL_SAMPLES  : {num_samples}")
    print("=" * 60)

    if not hf_token:
        print("[ERROR] HF_TOKEN が未設定です。Vertex AI の env またはシークレットマネージャで設定してください。")
        sys.exit(1)

    import sys as _sys
    _sys.argv = [
        "real_evaluate.py",
        "--mode",           mode,
        "--quant",          quant,
        "--quant_backend",  quant_backend,
        "--num_samples",    str(num_samples),
        "--finetune_steps", str(finetune_steps),
        "--student_model",  student_model,
        "--lora_r",         str(lora_r),
        "--target_domain",  target_domain,
        "--hf_token",       hf_token,
    ]

    try:
        from real_evaluate import main as eval_main
        results = eval_main()
        exit_code = 0
    except Exception:
        traceback.print_exc()
        exit_code = 1
        results = None

    # GCS アップロード
    if aip_model_dir:
        from real_evaluate import upload_to_gcs, RESULTS_DIR
        upload_to_gcs(RESULTS_DIR, aip_model_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
