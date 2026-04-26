# ============================================================
# Makefile — UniAD + WorldEngine Docker 操作ショートカット
# ============================================================
.PHONY: help build-all build-uniad2 build-algengine build-simengine \
        up-uniad2 up-algengine up-simengine up-all \
        train-stage1 train-stage2 train-rl eval-openloop eval-closedloop \
        rollout extract-rare tensorboard down clean

COMPOSE = docker compose
ENV_FILE = .env

help:
	@echo "=== UniAD + WorldEngine Docker コマンド ==="
	@echo ""
	@echo "【ビルド】"
	@echo "  make build-all          全イメージをビルド"
	@echo "  make build-uniad2       UniAD 事前学習イメージのみ"
	@echo "  make build-algengine    AlgEngine イメージのみ"
	@echo "  make build-simengine    SimEngine イメージのみ"
	@echo ""
	@echo "【起動】"
	@echo "  make up-algengine       AlgEngine コンテナ起動"
	@echo "  make up-simengine       SimEngine HEAD + WORKER 起動"
	@echo "  make up-all             全サービス起動"
	@echo "  make tensorboard        TensorBoard 起動 (port 6006)"
	@echo ""
	@echo "【学習・評価】"
	@echo "  make train-stage1       UniAD Stage1 (Perception) 学習"
	@echo "  make train-stage2       UniAD Stage2 (E2E) 学習"
	@echo "  make train-rl           AlgEngine RL ファインチューニング"
	@echo "  make eval-openloop      Open-loop 評価 (navtest)"
	@echo "  make eval-closedloop    Closed-loop 評価 (SimEngine)"
	@echo "  make rollout            SimEngine ロールアウト生成"
	@echo "  make extract-rare       希少ケース抽出"
	@echo ""
	@echo "【クリーンアップ】"
	@echo "  make down               全コンテナ停止"
	@echo "  make clean              コンテナ・イメージを削除"

# ────────────────────────────────────────────────────────────
# ビルド
# ────────────────────────────────────────────────────────────
build-all: build-uniad2 build-algengine build-simengine

build-uniad2:
	$(COMPOSE) --env-file $(ENV_FILE) build uniad2

build-algengine:
	$(COMPOSE) --env-file $(ENV_FILE) build algengine

build-simengine:
	$(COMPOSE) --env-file $(ENV_FILE) build simengine-head

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
	$(COMPOSE) --env-file $(ENV_FILE) up -d --scale simengine-worker=$(N) simengine-worker

# ────────────────────────────────────────────────────────────
# UniAD 学習
# ────────────────────────────────────────────────────────────
train-stage1:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage1_track_map/base_track_map.py \
	        8 \
	        --work-dir work_dirs/stage1_base"

train-stage2:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage2_e2e/base_e2e.py \
	        8 \
	        --work-dir work_dirs/stage2_e2e"

# ────────────────────────────────────────────────────────────
# AlgEngine 学習・評価
# ────────────────────────────────────────────────────────────
train-rl:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_train.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        8 \
	        work_dirs/e2e_uniad_50pct/epoch_20.pth"

eval-openloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_eval.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
	        8"

extract-rare:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    python scripts/rare_case_sampling_by_pdms.py \
	        --pdm-result work_dirs/e2e_uniad_50pct/navtest.csv \
	        --base-split configs/navsim_splits/navtest_split/navtest.yaml \
	        --output-dir configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare

# ────────────────────────────────────────────────────────────
# SimEngine ロールアウト & 閉ループ評価
# ────────────────────────────────────────────────────────────
rollout:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash scripts/run_ray_distributed_rollout.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct.py \
	        /workspace/WorldEngine/data/alg_engine/ckpts/e2e_uniad_50pct_ep20.pth \
	        e2e_uniad_50pct \
	        navtrain_50pct_collision \
	        navtrain

eval-closedloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
	        e2e_uniad_rlft \
	        navtest_failures \
	        NR

# ────────────────────────────────────────────────────────────
# クリーンアップ
# ────────────────────────────────────────────────────────────
down:
	$(COMPOSE) --env-file $(ENV_FILE) down

clean:
	$(COMPOSE) --env-file $(ENV_FILE) down --rmi local --volumes --remove-orphans
