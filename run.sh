#!/bin/bash
# ============================================================
# run.sh — MyAD / UniAD GCP 実行セットアップ
# ============================================================

# 1. gcloud 認証
gcloud auth login
gcloud config set project MyAD

# 2. 必要な API を有効化
gcloud services enable \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com

# 3. ジョブ投入
chmod +x submit_job.sh
./submit_job.sh

# タスクだけ変える場合 ---
# submit_job.sh の --args 部分を書き換えて再実行するだけ
#
# 例: Planning 評価のみ実行
#   EVAL_TASKS=planning ./submit_job.sh
#
# 例: 悪天候ロバスト性評価
#   EVAL_MODE=corruption ./submit_job.sh
