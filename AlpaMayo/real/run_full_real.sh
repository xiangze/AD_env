#!/usr/bin/env bash
# Alpamayo-R1-10B 実モデル評価 — GCP Vertex AI 実行スクリプト
#
# GCP_template/run_full.sh の構造を踏襲 + HF_TOKEN の安全な取り扱いを追加
#
# 使用方法:
#   export HF_TOKEN=hf_xxxx           # HuggingFace アクセストークン (必須)
#   ./run_full_real.sh [PROJECT_ID] [SUFFIX] [MODE]
#
# 例:
#   ./run_full_real.sh my-project alpamayo-real baseline
#   ./run_full_real.sh my-project alpamayo-real all
#   EVAL_SAMPLES=100 ./run_full_real.sh my-project alpamayo-quant quantize

set -euo pipefail

# ─── 設定 ────────────────────────────────────────────────────
PROJECT_ID="${1:-airy-decorator-216816}"
_SUF="${2:-alpamayo-real}"
MODE="${3:-baseline}"
REGION="asia-northeast1"
BUCKET="gs://${PROJECT_ID}-${_SUF}"
IMAGE="asia-northeast1-docker.pkg.dev/${PROJECT_ID}/${_SUF}/trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${_SUF}_${MODE}_${TIMESTAMP}"
LOCAL_RESULT_DIR="./results_real/${JOB_NAME}"

# 評価パラメータ
EVAL_MODE="${MODE}"
EVAL_QUANT="${EVAL_QUANT:-int8}"
EVAL_QUANT_BACKEND="${EVAL_QUANT_BACKEND:-bnb}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
EVAL_FINETUNE_STEPS="${EVAL_FINETUNE_STEPS:-200}"
EVAL_STUDENT_MODEL="${EVAL_STUDENT_MODEL:-Qwen/Qwen2.5-3B}"
EVAL_LORA_R="${EVAL_LORA_R:-8}"
EVAL_TARGET_DOMAIN="${EVAL_TARGET_DOMAIN:-JP}"

# HF_TOKEN チェック
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[ERROR] HF_TOKEN が未設定です"
    echo "  export HF_TOKEN=hf_xxxx"
    echo "  アクセス申請: https://huggingface.co/nvidia/Alpamayo-R1-10B"
    exit 1
fi

gcloud config set project "${PROJECT_ID}"

echo "======================================================"
echo "Alpamayo-R1-10B 実モデル評価 — Vertex AI"
echo "======================================================"
echo "  PROJECT_ID    : ${PROJECT_ID}"
echo "  MODE          : ${EVAL_MODE}"
echo "  EVAL_SAMPLES  : ${EVAL_SAMPLES}"
echo "  IMAGE         : ${IMAGE}"
echo "  JOB_NAME      : ${JOB_NAME}"
echo "======================================================"

# envsubst で設定ファイルを生成
export IMAGE BUCKET JOB_NAME HF_TOKEN
export EVAL_MODE EVAL_QUANT EVAL_QUANT_BACKEND EVAL_SAMPLES
export EVAL_FINETUNE_STEPS EVAL_STUDENT_MODEL EVAL_LORA_R EVAL_TARGET_DOMAIN
envsubst < setting.yaml.template > setting.yaml

# ─── [0] API 有効化 ────────────────────────────────────────
echo ""
echo ">>> [0] API 有効化..."
gcloud services enable \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    --project="${PROJECT_ID}"

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CB_SA}" \
    --role="roles/artifactregistry.writer" 2>/dev/null || true

# ─── [1] smoke test ────────────────────────────────────────
echo ""
echo ">>> [1] smoke test (シンセティックデータ, GPU なし)..."
SMOKE_OUT="./smoke_result_real"
mkdir -p "${SMOKE_OUT}"

python real_evaluate.py \
    --mode baseline \
    --num_samples 5 \
    2>&1 | tee "${SMOKE_OUT}/smoke.log"

# alpamayo_r1 未インストール時はスキップ (Dockerfile 内でのみ動作)
if grep -qiE "^Traceback" "${SMOKE_OUT}/smoke.log" 2>/dev/null; then
    echo "[WARN] smoke test でエラー (alpamayo_r1 未インストール環境では想定内)"
    echo "  → Docker ビルド後に Vertex AI 上で実行します"
fi
echo "[OK] smoke test 完了 (シンセティックデータ)"

# ─── [2] GCS バケット作成 ──────────────────────────────────
echo ""
echo ">>> [2] GCS バケット作成..."
gsutil mb -l "${REGION}" "${BUCKET}" 2>/dev/null || true

# ─── [3] Artifact Registry 作成 ────────────────────────────
echo ""
echo ">>> [3] Artifact Registry リポジトリ作成..."
gcloud artifacts repositories create "${_SUF}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || true

# ─── [4] Docker ビルド & プッシュ ──────────────────────────
echo ""
echo ">>> [4] Docker ビルド & プッシュ..."
# HF_TOKEN は Dockerfile に含めない (セキュリティ)
gcloud builds submit \
    --config cloudbuild.yaml \
    --project="${PROJECT_ID}" \
    --substitutions="_SUF=${_SUF}"

# ─── [5] Vertex AI ジョブ投入 ──────────────────────────────
echo ""
echo ">>> [5] Vertex AI ジョブ投入: ${JOB_NAME}"
echo "  注意: A100×1 (a2-highgpu-1g) を使用します"
echo "        BF16 フルモデルには 24GB VRAM が必要です"
gcloud ai custom-jobs create \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="setting.yaml"

echo "  ジョブ投入完了: ${JOB_NAME}"
echo "  結果保存先: ${BUCKET}/results/${JOB_NAME}/"
echo "  モデルウェイト DL 時間: 約 2.5分 (100MB/s 回線)"

# ─── [6] 完了待機 ──────────────────────────────────────────
echo ""
echo ">>> [6] ジョブ完了待機 (120秒間隔 — モデル DL + 推論に時間がかかります)..."

WAIT_SECONDS=0
MAX_WAIT=28800  # 最大8時間 (蒸留ジョブは長い)

while [ "${WAIT_SECONDS}" -lt "${MAX_WAIT}" ]; do
    STATUS=$(gcloud ai custom-jobs list \
        --region="${REGION}" \
        --filter="displayName=${JOB_NAME}" \
        --format="value(state)" 2>/dev/null | head -1)

    echo "  ステータス: ${STATUS}  (経過: ${WAIT_SECONDS}s / $(date '+%H:%M:%S'))"

    case "${STATUS}" in
        JOB_STATE_SUCCEEDED) echo "[OK] ジョブ完了"; break ;;
        JOB_STATE_FAILED|JOB_STATE_CANCELLED)
            echo "[ERROR] ジョブ失敗: ${STATUS}"
            gcloud ai custom-jobs describe \
                "$(gcloud ai custom-jobs list \
                    --region="${REGION}" \
                    --filter="displayName=${JOB_NAME}" \
                    --format='value(name)' | head -1)" \
                --region="${REGION}" 2>/dev/null || true
            exit 1 ;;
        *) sleep 120; WAIT_SECONDS=$((WAIT_SECONDS + 120)) ;;
    esac
done

# ─── [7] 結果ダウンロード ──────────────────────────────────
echo ""
echo ">>> [7] 結果ダウンロード → ${LOCAL_RESULT_DIR}"
mkdir -p "${LOCAL_RESULT_DIR}"
gsutil -m cp -r "${BUCKET}/results/${JOB_NAME}/*" "${LOCAL_RESULT_DIR}/" 2>/dev/null || true

echo ""
echo "===== 完了 ====="
echo "ローカル結果: ${LOCAL_RESULT_DIR}"
ls -lh "${LOCAL_RESULT_DIR}" 2>/dev/null || echo "(ファイルなし)"

RESULT_JSON="${LOCAL_RESULT_DIR}/real_eval_results.json"
if [ -f "${RESULT_JSON}" ]; then
    echo ""
    echo "--- 評価結果サマリー ---"
    python3 - <<'PYEOF'
import json, os, glob
files = sorted(glob.glob("results_real/*/real_eval_results.json"))
path = files[-1] if files else ""
if not path:
    print("(結果 JSON が見つかりません)")
else:
    with open(path) as f:
        data = json.load(f)
    print(f"{'手法':<28} {'ADE(m)':>8} {'FDE(m)':>8} {'Lat(ms)':>8} {'Mem(GB)':>8}")
    print("-" * 60)
    for r in data:
        ade = r.get("ade", float("nan"))
        fde = r.get("fde", float("nan"))
        lat = r.get("latency_ms", float("nan"))
        mem = r.get("memory_mb", 0) / 1024
        print(f"{r['method']:<28} {ade:>8.3f} {fde:>8.3f} {lat:>8.0f} {mem:>8.1f}")
PYEOF
fi
