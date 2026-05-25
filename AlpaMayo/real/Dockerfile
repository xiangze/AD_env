# Alpamayo-R1-10B 実モデル評価 — Vertex AI A100 用 Docker イメージ
#
# ハードウェア要件:
#   - NVIDIA A100 40GB または 80GB (VRAM ≥ 24GB 必須)
#   - Vertex AI: a2-highgpu-1g (A100×1) または a2-highgpu-2g (A100×2)
#
# ソフトウェア要件 (Alpamayo-R1 公式):
#   - PyTorch >= 2.8
#   - Transformers >= 4.57.1  (Qwen3VL サポート)
#   - DeepSpeed >= 0.17.4     (大規模モデルの分散推論)
#   - CUDA 12.1+
#
# ライセンス:
#   - モデルウェイト: 非商用ライセンス
#   - 推論コード (alpamayo_r1): Apache 2.0

FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# ─── システム依存 ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# ─── uv のインストール (Alpamayo-R1 公式推奨) ───────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ─── Alpamayo-R1 パッケージのインストール ────────────────────
# 公式: uv venv && uv sync
# Docker 内では pip 経由でインストール
RUN pip install --no-cache-dir \
    "torch>=2.8.0" \
    "transformers>=4.57.1" \
    "deepspeed>=0.17.4" \
    "accelerate>=0.28.0" \
    "peft>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "torchao>=0.4.0" \
    "einops" \
    "sentencepiece" \
    "pillow" \
    "numpy" \
    "scipy" \
    "google-cloud-storage>=2.14.0" \
    "huggingface_hub>=0.22.0" \
    "scikit-learn>=1.4.0" \
    "tqdm"

# Alpamayo-R1 公式パッケージ (GitHub から直接インストール)
RUN pip install --no-cache-dir \
    "git+https://github.com/NVlabs/alpamayo.git"

# PhysicalAI-AV データセットのローダー
RUN pip install --no-cache-dir \
    "git+https://github.com/NVlabs/physical_ai_av.git" 2>/dev/null || \
    echo "[WARN] physical_ai_av のインストールに失敗 (シンセティックデータを使用)"

# ─── アプリケーションコード ─────────────────────────────────
COPY real_evaluate.py .
COPY trainer/         trainer/

# HuggingFace キャッシュ用ディレクトリ
# (モデルウェイト ~22GB が /tmp/hf_cache にキャッシュされる)
RUN mkdir -p /tmp/hf_cache /app/results_real
ENV HF_HOME="/tmp/hf_cache"
ENV TRANSFORMERS_CACHE="/tmp/hf_cache"

# Flash Attention 2 (A100/H100 で推論高速化)
# オプション: インストールに時間がかかるため別 stage に分離も可
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "[INFO] flash-attn のインストールをスキップ (標準 attention を使用)"

# ─── エントリーポイント ─────────────────────────────────────
# GCP_template と同じ trainer.task 構造
ENTRYPOINT ["python", "-m", "trainer.task"]
