#!/usr/bin/env bash
# =============================================================================
# prepare_mini_data.sh
# nuScenes v1.0-mini による UniAD 評価の「配管」を可能な範囲で自動化する。
#
# 自動化できないもの(手動で用意):
#   ログイン必須のため、以下を $DOWNLOADS に置いておくこと:
#     - v1.0-mini.tgz       (nuScenes mini 本体)
#     - can_bus.zip         (CAN bus expansion)
#
# 自動化されるもの(すべて冪等: 既にあればスキップ):
#   - ディレクトリ作成
#   - アーカイブ展開(mini 本体 / can_bus)
#   - motion anchor 取得(公開 release)
#   - create_data.py による mini info pkl 生成(コンテナ内)
#
# 使い方(docker-compose.yml のあるディレクトリで):
#   DATA_ROOT=/mnt/data ./prepare_mini_data.sh
# =============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/data}"
DOWNLOADS="${DOWNLOADS:-$DATA_ROOT/downloads}"
NUSC="$DATA_ROOT/nuscenes"
INFOS="$DATA_ROOT/infos"
OTHERS="$DATA_ROOT/others"
COMPOSE="${COMPOSE:-docker compose}"
ENV_FILE="${ENV_FILE:-.env}"
ANCHOR_URL="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl"

log()  { printf '\033[1;34m[prep]\033[0m %s\n' "$*"; }
skip() { printf '\033[1;32m[skip]\033[0m %s\n' "$*"; }

# --- 0. ディレクトリ ---------------------------------------------------------
mkdir -p "$NUSC" "$INFOS" "$OTHERS" "$DOWNLOADS"

# --- 1. nuScenes mini 本体の展開 --------------------------------------------
if [ -f "$NUSC/v1.0-mini/scene.json" ]; then
  skip "v1.0-mini already extracted"
else
  [ -f "$DOWNLOADS/v1.0-mini.tgz" ] || {
    echo "ERROR: $DOWNLOADS/v1.0-mini.tgz が見つかりません。" \
         "nuScenes からログインして取得し、そこへ置いてください。" >&2
    exit 1
  }
  log "extracting v1.0-mini.tgz -> $NUSC"
  tar -xzf "$DOWNLOADS/v1.0-mini.tgz" -C "$NUSC"
fi

# --- 2. can_bus の展開 -------------------------------------------------------
if ls "$NUSC/can_bus"/*.json >/dev/null 2>&1; then
  skip "can_bus already extracted"
else
  [ -f "$DOWNLOADS/can_bus.zip" ] || {
    echo "ERROR: $DOWNLOADS/can_bus.zip が見つかりません。" \
         "nuScenes の CAN bus expansion を取得して置いてください。" >&2
    exit 1
  }
  log "extracting can_bus.zip -> $NUSC"
  unzip -q -o "$DOWNLOADS/can_bus.zip" -d "$NUSC"
fi

# --- 3. motion anchor(公開 release なので自動取得可) -----------------------
if [ -f "$OTHERS/motion_anchor_infos_mode6.pkl" ]; then
  skip "motion anchor already present"
else
  log "downloading motion_anchor_infos_mode6.pkl"
  wget -q -O "$OTHERS/motion_anchor_infos_mode6.pkl" "$ANCHOR_URL"
fi

# --- 4. mini info pkl 生成(コンテナ内) ------------------------------------
if [ -f "$INFOS/nuscenes_infos_temporal_val.pkl" ]; then
  VER=$($COMPOSE --env-file "$ENV_FILE" exec -T uniad2 python -c \
    "import pickle;print(pickle.load(open('data/infos/nuscenes_infos_temporal_val.pkl','rb'))['metadata']['version'])" \
    2>/dev/null | tr -d '\r' || echo "unknown")
  if [ "$VER" = "v1.0-mini" ]; then
    skip "mini info pkl already present (version=$VER)"
    log "done. now run:  make eval-uniad-openloop"
    exit 0
  fi
  log "existing pkl is '$VER' -> backing up and regenerating for mini"
  mkdir -p "$INFOS/_backup"
  mv "$INFOS"/nuscenes_infos_temporal_*.pkl "$INFOS/_backup/" 2>/dev/null || true
fi

log "starting uniad2 and generating mini info pkl"
$COMPOSE --env-file "$ENV_FILE" up -d uniad2
$COMPOSE --env-file "$ENV_FILE" exec -T uniad2 bash -c \
  "cd /workspace/UniAD && python tools/create_data.py nuscenes \
     --root-path ./data/nuscenes --out-dir ./data/infos \
     --extra-tag nuscenes --version v1.0-mini --canbus ./data/nuscenes"

log "verifying"
$COMPOSE --env-file "$ENV_FILE" exec -T uniad2 python -c \
  "import pickle; d=pickle.load(open('data/infos/nuscenes_infos_temporal_val.pkl','rb')); \
   print('version:', d['metadata']['version'], ' infos:', len(d['infos']))"

log "done. now run:  make eval-uniad-openloop"
