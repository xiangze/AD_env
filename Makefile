# ============================================================
# Makefile — UniAD + WorldEngine Docker ビルド・運用
#
# ビルド順序:
#   ① base       → 共有レイヤ (APT / Python / PyTorch)
#   ② uniad2     → base を継承 (mmcv wheel + UniAD)
#   ③ algengine  → base を継承 (mmcv src build + WorldEngine)
#   ④ simengine  → base を継承 (gsplat + Ray + SimEngine)
#
# ② ③ ④ は base さえできれば並列ビルド可能
# ============================================================
.PHONY: help build-base build-uniad2 build-algengine build-simengine \
        build-all build-parallel \
        up-uniad2 up-algengine up-simengine up-all \
        train-stage1 train-stage2 train-rl \
        eval-openloop eval-closedloop eval-uniad-openloop\
        rollout extract-rare tensorboard \
        down clean

COMPOSE   = docker compose
ENV_FILE  = .env
BASE_TAG  = uniad-worldengine-base:latest

help:
	@echo "=== UniAD + WorldEngine Docker ==="
	@echo ""
	@echo "【ビルド】"
	@echo "  make build-base       ① ベースイメージ (必ず最初に実行)"
	@echo "  make build-all        ①〜④ を順番にビルド"
	@echo "  make build-parallel   ①の後に②③④を並列ビルド (make -j3)"
	@echo "  make build-uniad2     ② UniAD 事前学習イメージ"
	@echo "  make build-algengine  ③ AlgEngine RL イメージ"
	@echo "  make build-simengine  ④ SimEngine 3DGS イメージ"
	@echo ""
	@echo "【起動】"
	@echo "  make up-uniad2        UniAD コンテナ起動"
	@echo "  make up-algengine     AlgEngine コンテナ起動"
	@echo "  make up-simengine     SimEngine HEAD + WORKER 起動"
	@echo "  make tensorboard      TensorBoard (port 6006)"
	@echo "  make build-mmcv-ops-uniad2    mmcv CUDA op ビルド (uniad2コンテナ)"
	@echo "  make build-mmcv-ops-algengine mmcv CUDA op ビルド (algengineコンテナ)"
	@echo ""
	@echo "【学習・評価】"
	@echo "  make train-stage1     UniAD Stage1 Perception 学習"
	@echo "  make train-stage2     UniAD Stage2 E2E 学習"
	@echo "  make train-rl         AlgEngine RL ファインチューニング"
	@echo "  make rollout          SimEngine ロールアウト生成"
	@echo "  make extract-rare     希少ケース抽出"
	@echo "  make eval-openloop    Open-loop 評価"
	@echo "  make eval-uniad-openloop Open-loop 評価(UniAD)"
	@echo "  make eval-closedloop  Closed-loop 評価"
	@echo ""
	@echo "【クリーンアップ】"
	@echo "  make down             全コンテナ停止"
	@echo "  make clean            コンテナ・イメージを削除"

# ────────────────────────────────────────────────────────────
# ビルド — 順序が重要
# ────────────────────────────────────────────────────────────

# ① ベースイメージ (他の全イメージが依存するため必ず最初に実行)
build-base:
	docker build \
	    -f Dockerfile.base \
	    -t $(BASE_TAG) \
	    .
	@echo "✓ base image built: $(BASE_TAG)"

#build-worldengine-base:
#	docker build -f Dockerfile.base -t uniad-worldengine-base:latest .

# ② UniAD 事前学習 (build-base が完了してから実行)
build-uniad2: build-base
	docker build \
	    -f Dockerfile.uniad2 \
	    -t uniad2:latest \
	    .

# ③ AlgEngine RL (build-base が完了してから実行)
build-algengine: build-base
	docker build \
	    -f Dockerfile.algengine \
	    -t algengine:latest \
	    .

# ④ SimEngine 3DGS (build-base が完了してから実行)
build-simengine: build-base
	docker build \
	    -f Dockerfile.simengine \
	    -t simengine:latest \
	    .

# 全イメージを順番にビルド (安全な直列実行)
build-all: build-base build-uniad2 build-algengine build-simengine
	@echo "✓ All images built."

# ② ③ ④ を並列ビルド (base 完了後に make -j3 で実行)
# 使い方: make build-base && make build-parallel -j3
build-parallel: build-uniad2 build-algengine build-simengine

# ────────────────────────────────────────────────────────────
# 起動
# ────────────────────────────────────────────────────────────
up-uniad2:
	$(COMPOSE) --env-file $(ENV_FILE) up -d uniad2

up-algengine:
	$(COMPOSE) --env-file $(ENV_FILE) up -d algengine

up-simengine:
	$(COMPOSE) --env-file $(ENV_FILE) up -d simengine-head simengine-worker

up-all:
	$(COMPOSE) --env-file $(ENV_FILE) up -d

tensorboard:
	$(COMPOSE) --env-file $(ENV_FILE) up -d tensorboard
	@echo "TensorBoard: http://localhost:6006"

scale-workers:
	$(COMPOSE) --env-file $(ENV_FILE) \
	    up -d --scale simengine-worker=$(N) simengine-worker

# ────────────────────────────────────────────────────────────
# 学習・評価
# ────────────────────────────────────────────────────────────
train-stage1:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage1_track_map/base_track_map.py 8 \
	        --work-dir work_dirs/stage1_base"

train-stage2:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage2_e2e/base_e2e.py 8 \
	        --work-dir work_dirs/stage2_e2e"

train-rl:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_train.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py 8 \
	        work_dirs/e2e_uniad_50pct/epoch_20.pth"

rollout:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash scripts/run_ray_distributed_rollout.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct.py \
	        /workspace/WorldEngine/data/alg_engine/ckpts/e2e_uniad_50pct_ep20.pth \
	        e2e_uniad_50pct \
	        navtrain_50pct_collision \
	        navtrain

extract-rare:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    python scripts/rare_case_sampling_by_pdms.py \
	        --pdm-result work_dirs/e2e_uniad_50pct/navtest.csv \
	        --base-split configs/navsim_splits/navtest_split/navtest.yaml \
	        --output-dir configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare

# ────────────────────────────────────────────────────────────
# open loop精度評価
# ────────────────────────────────────────────────────────────
eval-uniad-openloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	  bash -c "cd /workspace/UniAD && \
	    ./tools/uniad_dist_eval.sh $(UNIAD_EVAL_CFG) $(UNIAD_EVAL_CKPT) $(UNIAD_EVAL_GPUS) " 

eval-openloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_eval.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth 1"

eval-closedloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
	        e2e_uniad_rlft navtest_failures NR

# ────────────────────────────────────────────────────────────
# クリーンアップ
# ────────────────────────────────────────────────────────────
down:
	$(COMPOSE) --env-file $(ENV_FILE) down

clean:
	$(COMPOSE) --env-file $(ENV_FILE) down --rmi local --volumes --remove-orphans
	docker rmi $(BASE_TAG) 2>/dev/null || true

# ────────────────────────────────────────────────────────────
# mmcv CUDA op 後ビルド (初回コンテナ起動後に一度だけ実行)
# ────────────────────────────────────────────────────────────
build-mmcv-ops-uniad2:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    build_mmcv_ops.sh

build-mmcv-ops-algengine:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    build_mmcv_ops.sh

build-mmcv-ops-all: build-mmcv-ops-uniad2 build-mmcv-ops-algengine
	@echo "✓ mmcv CUDA ops built in all containers."

# ────────────────────────────────────────────────────────────
# 軽量化 (Compression)
# ────────────────────────────────────────────────────────────
COMPRESSION_CKPT  ?= ckpts/uniad_base_e2e.pth
COMPRESSION_CFG   ?= projects/configs/stage2_e2e/base_e2e.py
VRAM_BUDGET       ?= 12

# FP16 量子化
compress-fp16:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/compression/quantize.py \
	        --config $(COMPRESSION_CFG) \
	        --ckpt   $(COMPRESSION_CKPT) \
	        --mode   fp16 \
	        --out    ckpts/uniad_fp16.pth

# INT8 動的量子化
compress-int8:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/compression/quantize.py \
	        --config $(COMPRESSION_CFG) \
	        --ckpt   $(COMPRESSION_CKPT) \
	        --mode   int8 \
	        --out    ckpts/uniad_int8.pth

# 軽量 config 生成 (VRAM_BUDGET GB 向け)
make-lite-config:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/compression/make_lite_config.py \
	        --base-config $(COMPRESSION_CFG) \
	        --vram-budget $(VRAM_BUDGET) \
	        --out-dir     projects/configs/stage2_e2e/

# 全 VRAM プロファイルの config を一括生成
make-lite-config-all:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/compression/make_lite_config.py \
	        --base-config $(COMPRESSION_CFG) \
	        --out-dir     projects/configs/stage2_e2e/ \
	        --all

# ────────────────────────────────────────────────────────────
# ファインチューニング (Fine-tuning)
# ────────────────────────────────────────────────────────────
FT_CKPT    ?= ckpts/uniad_base_e2e.pth
FT_CFG     ?= projects/configs/stage2_e2e/base_e2e.py
FT_DATA    ?= data/custom/infos_train.pkl
FT_GPUS    ?= 4
FT_EPOCHS  ?= 8
FT_VRAM    ?= 40
FT_OUT     ?= work_dirs/finetune_$(shell date +%Y%m%d_%H%M%S)

# データ前処理 (nuScenes / ACDC / カスタム CSV)
prepare-data-nuscenes:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/finetune/prepare_custom_data.py \
	        --input-format nuscenes \
	        --data-root    data/nuscenes \
	        --out-dir      data/custom \
	        --split        train

prepare-data-acdc:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/finetune/prepare_custom_data.py \
	        --input-format acdc \
	        --data-root    data/acdc \
	        --conditions   fog rain night \
	        --out-dir      data/custom_acdc \
	        --split        train

# ドメイン適応 FT (Planning Head のみ, 軽量)
finetune-domain:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/finetune/finetune.py \
	        --mode    domain \
	        --config  $(FT_CFG) \
	        --ckpt    $(FT_CKPT) \
	        --data    $(FT_DATA) \
	        --gpus    $(FT_GPUS) \
	        --epochs  $(FT_EPOCHS) \
	        --vram    $(FT_VRAM) \
	        --out-dir $(FT_OUT)

# タスク特化 FT (BEV Encoder も更新)
finetune-task:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/finetune/finetune.py \
	        --mode    task \
	        --config  $(FT_CFG) \
	        --ckpt    $(FT_CKPT) \
	        --data    $(FT_DATA) \
	        --gpus    $(FT_GPUS) \
	        --epochs  20 \
	        --vram    $(FT_VRAM) \
	        --out-dir $(FT_OUT)

# ────────────────────────────────────────────────────────────
# GCP ジョブ投入 (Python ベース)
# ────────────────────────────────────────────────────────────
gcp-eval:
	python3 tools/gcp/submit_job.py \
	    --job-type eval \
	    --tasks    $(EVAL_TASKS) \
	    --ckpt     $(GCS_CKPT) \
	    --gpus     $(N_GPUS)

gcp-finetune:
	python3 tools/gcp/submit_job.py \
	    --job-type finetune \
	    --ft-mode  $(FT_MODE) \
	    --data     $(GCS_DATA) \
	    --gpus     $(N_GPUS)

gcp-rl:
	python3 tools/gcp/submit_job.py \
	    --job-type rl \
	    --gpus     8

# ────────────────────────────────────────────────────────────
# VAD (Vectorized Autonomous Driving)
# 注意: UniAD とは独立した Python 3.8 / CUDA 11.1 / torch 1.9 環境
# Dockerfile.vad は Dockerfile.base に依存しない (独自 FROM)
# ────────────────────────────────────────────────────────────
VAD_CONFIG ?= projects/configs/VAD/VAD_base_stage_2.py
VAD_CKPT   ?= ckpts/VAD_base.pth
VAD_OUT    ?= eval_results/vad_$(shell date +%Y%m%d_%H%M%S)
VAD_CONFIG ?= projects/configs/VAD/VAD_base_e2e.py   # ls projects/configs/VAD/ で実名確認
VAD_CKPT   ?= ckpts/VAD_base.pth

build-vad:
	docker build -f Dockerfile.vad -t vad:latest .
	@echo "✓ VAD image built: vad:latest"
# VAD コンテナ起動
up-vad:
	$(COMPOSE) --env-file $(ENV_FILE) up -d vad

# nuScenes 用 VAD 専用 info pkl を生成(初回のみ。data/nuscenes へ書き込む)
prepare-vad-data:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad bash -c "cd /workspace/VAD && \
	  python tools/data_converter/vad_nuscenes_converter.py nuscenes \
	    --root-path ./data/nuscenes --out-dir ./data/nuscenes \
	    --extra-tag vad_nuscenes --version v1.0-trainval --canbus ./data"

# 評価は単一 GPU・非分散(VAD 公式の指定)
eval-vad:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad bash -c "cd /workspace/VAD && \
	  CUDA_VISIBLE_DEVICES=0 python tools/test.py $(VAD_CONFIG) $(VAD_CKPT) \
	    --launcher none --eval bbox --tmpdir tmp"

eval-vad-tiny:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad \
	    python3 /workspace/VAD/eval_scripts/eval_vad.py \
	        --config  projects/configs/VAD/VAD_tiny_stage_2.py \
	        --ckpt    ckpts/VAD_tiny.pth \
	        --out-dir $(VAD_OUT) \
	        --variant tiny

# VAD-Tiny 学習 (8 GPU)
train-vad-tiny:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad \
	    python3 -m torch.distributed.run \
	        --nproc_per_node=8 \
	        --master_port=2333 \
	        tools/train.py \
	        projects/configs/VAD/VAD_tiny_stage_2.py \
	        --launcher pytorch \
	        --deterministic \
	        --work-dir work_dirs/vad_tiny

# VAD-Base 学習 (8 GPU)
train-vad-base:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad \
	    python3 -m torch.distributed.run \
	        --nproc_per_node=8 \
	        --master_port=2333 \
	        tools/train.py \
	        projects/configs/VAD/VAD_base_stage_2.py \
	        --launcher pytorch \
	        --deterministic \
	        --work-dir work_dirs/vad_base

# ────────────────────────────────────────────────────────────
# 量子化・蒸留後の精度評価
# ────────────────────────────────────────────────────────────
COMP_CONFIG  ?= projects/configs/stage2_e2e/base_e2e.py
COMP_OUT     ?= eval_results/compression_$(shell date +%Y%m%d_%H%M%S)

# UniAD: FP32 / FP16 / INT8 の精度比較
eval-compression-uniad:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 eval_scripts/eval_compressed.py \
	        --model   uniad \
	        --config  $(COMP_CONFIG) \
	        --ckpts   ckpts/uniad_base_e2e.pth:fp32 \
	                  ckpts/uniad_fp16.pth:fp16 \
	                  ckpts/uniad_int8.pth:int8 \
	        --tasks   planning,tracking \
	        --out-dir $(COMP_OUT)/uniad

# VAD: FP32 / FP16 の精度比較
eval-compression-vad:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad \
	    python3 eval_scripts/eval_compressed.py \
	        --model    vad \
	        --variant  base \
	        --config   projects/configs/VAD/VAD_base_stage_2.py \
	        --ckpts    ckpts/VAD_base.pth:fp32 \
	                   ckpts/VAD_base_fp16.pth:fp16 \
	        --out-dir  $(COMP_OUT)/vad

# UniAD: Teacher(R101) → Student(R50) 蒸留
distill-uniad:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    python3 tools/compression/distill.py \
	        --model           uniad \
	        --teacher-config  projects/configs/stage2_e2e/base_e2e.py \
	        --teacher-ckpt    ckpts/uniad_base_e2e.pth \
	        --student-config  projects/configs/stage2_e2e/base_e2e_r50.py \
	        --mode            response \
	        --gpus            8 \
	        --epochs          10 \
	        --out-dir         work_dirs/distill_r101_to_r50

# VAD: Base → Tiny 蒸留
distill-vad:
	$(COMPOSE) --env-file $(ENV_FILE) exec vad \
	    python3 tools/compression/distill.py \
	        --model           vad \
	        --teacher-config  projects/configs/VAD/VAD_base_stage_2.py \
	        --teacher-ckpt    ckpts/VAD_base.pth \
	        --student-config  projects/configs/VAD/VAD_tiny_stage_2.py \
	        --mode            response \
	        --gpus            4 \
	        --epochs          20 \
	        --out-dir         work_dirs/distill_vad_base_to_tiny

# 全モデルの量子化 + 評価を一括実行
eval-all-compression:
	$(MAKE) compress-fp16
	$(MAKE) compress-int8
	$(MAKE) eval-compression-uniad
	$(MAKE) eval-compression-vad
