#!/bin/bash
# ============================================================
# submit_job.sh — MyAD / UniAD Vertex AI ジョブ投入
#
# 環境変数でタスクを切り替えられる:
#   EVAL_TASKS=all          全5タスク評価 (デフォルト)
#   EVAL_TASKS=planning     Planning のみ
#   EVAL_TASKS=tracking,mapping  複数指定
#   EVAL_MODE=corruption    nuScenes-C ロバスト性評価
#   N_GPUS=8                マルチGPU (Stage1/2 学習時)
# ============================================================
set -e

# ── 設定 ────────────────────────────────────────────────────
SUF=myad
PROJECT_ID=$(gcloud config get-value project)   # → MyAD
REGION="asia-northeast1"
BUCKET="gs://${PROJECT_ID}-${SUF}"
IMAGE="asia-northeast1-docker.pkg.dev/${PROJECT_ID}/${SUF}/trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${SUF}_${TIMESTAMP}"

# タスク設定 (環境変数で上書き可)
EVAL_TASKS="${EVAL_TASKS:-all}"
EVAL_MODE="${EVAL_MODE:-normal}"    # normal | corruption
N_GPUS="${N_GPUS:-1}"
CKPT="${CKPT:-/gcs/ckpts/uniad_base_e2e.pth}"
CONFIG="${CONFIG:-projects/configs/stage2_e2e/base_e2e.py}"

# ── GPU スペック選択 ─────────────────────────────────────────
# UniAD 評価: A100 40GB × 1 以上が必要 (T4 16GB は VRAM 不足)
# Stage1/2 学習: A100 × 8 推奨
if [ "$N_GPUS" -ge 8 ]; then
    MACHINE_TYPE="a2-highgpu-8g"
    ACCEL_TYPE="NVIDIA_TESLA_A100"
    ACCEL_COUNT=8
elif [ "$N_GPUS" -ge 2 ]; then
    MACHINE_TYPE="a2-highgpu-2g"
    ACCEL_TYPE="NVIDIA_TESLA_A100"
    ACCEL_COUNT=2
else
    # 評価専用: A100 × 1
    MACHINE_TYPE="a2-highgpu-1g"
    ACCEL_TYPE="NVIDIA_TESLA_A100"
    ACCEL_COUNT=1
fi

echo "=== MyAD Vertex AI ジョブ設定 ==="
echo "  Project   : ${PROJECT_ID}"
echo "  Job Name  : ${JOB_NAME}"
echo "  Machine   : ${MACHINE_TYPE} (${ACCEL_TYPE} × ${ACCEL_COUNT})"
echo "  Tasks     : ${EVAL_TASKS}"
echo "  Mode      : ${EVAL_MODE}"
echo "  Bucket    : ${BUCKET}"
echo "=================================="

# --- 1. GCS バケット作成 (初回のみ) ---
gsutil mb -l ${REGION} ${BUCKET} 2>/dev/null || true

# --- 2. nuScenes データとチェックポイントの GCS パスを確認 ---
# 事前に以下のように GCS へアップロードしておく:
#   gsutil -m cp -r /path/to/nuscenes/data   ${BUCKET}/data/nuscenes/
#   gsutil -m cp -r /path/to/ckpts/          ${BUCKET}/ckpts/
#   gsutil -m cp -r /path/to/data/infos/     ${BUCKET}/data/infos/

# --- 3. Artifact Registry リポジトリ作成 (初回のみ) ---
gcloud artifacts repositories create ${SUF} \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID} 2>/dev/null || true

# --- 4. Docker イメージをビルド & プッシュ ---
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_SUF=${SUF} \
  --project=${PROJECT_ID}

# --- 5. Vertex AI ジョブ投入 ---
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec="\
machine-type=${MACHINE_TYPE},\
accelerator-type=${ACCEL_TYPE},\
accelerator-count=${ACCEL_COUNT},\
container-image-uri=${IMAGE}" \
  --environment-variables="\
AIP_MODEL_DIR=${BUCKET}/results/${JOB_NAME},\
EVAL_TASKS=${EVAL_TASKS},\
EVAL_MODE=${EVAL_MODE},\
CKPT=${CKPT},\
CONFIG=${CONFIG},\
N_GPUS=${ACCEL_COUNT},\
GCS_DATA_ROOT=${BUCKET}/data,\
GCS_CKPT_ROOT=${BUCKET}/ckpts"

echo ""
echo "ジョブ投入完了: ${JOB_NAME}"
echo "結果保存先   : ${BUCKET}/results/${JOB_NAME}/"
echo ""
echo "ステータス確認:"
echo "  gcloud ai custom-jobs list --region=${REGION}"
echo "  gcloud ai custom-jobs describe <JOB_ID> --region=${REGION}"
echo ""
echo "ログ確認:"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION}"
