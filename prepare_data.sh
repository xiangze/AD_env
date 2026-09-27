#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh  —  nuScenes 生データを共有しつつ、モデル別 info を生成する
#
# 使い方(docker-compose.yml のあるディレクトリで):
#   DATA_ROOT=/mnt/data ./prepare_data.sh raw          # 生データ展開(1回だけ)
#   DATA_ROOT=/mnt/data ./prepare_data.sh uniad        # UniAD 用 info
#   DATA_ROOT=/mnt/data ./prepare_data.sh vad          # VAD 用 info
#   DATA_ROOT=/mnt/data ./prepare_data.sh sparsedrive  # SparseDrive 用 info+anchor
#   DATA_ROOT=/mnt/data ./prepare_data.sh all          # raw + 3モデル全部
#
# 手動で用意するもの($DOWNLOADS に置く。nuScenes はログイン必須):
#   - v1.0-mini.tgz
#   - can_bus.zip
#   - nuScenes-map-expansion-v1.3.zip
#
# 生データ(samples/sweeps/maps/can_bus/v1.0-mini)は全モデル共通。
# info pkl だけがモデル固有で、フォーマット非互換。モデルごとに別ディレクトリへ出す。
# =============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/data}"
DOWNLOADS="${DOWNLOADS:-$DATA_ROOT/downloads}"
NUSC="$DATA_ROOT/nuscenes"                       # 共有生データ(全モデル ro マウント)
COMPOSE="${COMPOSE:-docker compose}"
ENV_FILE="${ENV_FILE:-.env}"
ANCHOR_URL="https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl"
VERSION="${VERSION:-v1.0-mini}"                  # trainval に切替可

log()  { printf '\033[1;34m[prep]\033[0m %s\n' "$*"; }
skip() { printf '\033[1;32m[skip]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err ]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 生データ展開(全モデル共通・冪等)
# ---------------------------------------------------------------------------
prepare_raw() {
  mkdir -p "$NUSC" "$DOWNLOADS"

  if [ -f "$NUSC/$VERSION/scene.json" ]; then
    skip "$VERSION already extracted"
  else
    [ -f "$DOWNLOADS/$VERSION.tgz" ] || err "$DOWNLOADS/$VERSION.tgz が無い(nuScenes からログインして取得)"
    log "extracting $VERSION.tgz"
    tar -xzf "$DOWNLOADS/$VERSION.tgz" -C "$NUSC"
  fi

  if ls "$NUSC/can_bus"/*.json >/dev/null 2>&1; then
    skip "can_bus already extracted"
  else
    [ -f "$DOWNLOADS/can_bus.zip" ] || err "$DOWNLOADS/can_bus.zip が無い(CAN bus expansion)"
    log "extracting can_bus.zip"
    unzip -q -o "$DOWNLOADS/can_bus.zip" -d "$NUSC"
  fi

  if [ -f "$NUSC/maps/expansion/boston-seaport.json" ]; then
    skip "map expansion already extracted"
  else
    local mapzip
    mapzip=$(ls "$DOWNLOADS"/nuScenes-map-expansion-*.zip 2>/dev/null | head -1 || true)
    [ -n "$mapzip" ] || err "$DOWNLOADS に nuScenes-map-expansion-*.zip が無い(Map expansion)"
    log "extracting $(basename "$mapzip")"
    unzip -q -o "$mapzip" -d "$NUSC/maps"
  fi
  log "raw data ready at $NUSC"
}

# 共通: コンテナ内で pkl の version を確認して mini/trainval を判定
_pkl_version() {  # $1=service  $2=pkl path (container 相対)
  $COMPOSE --env-file "$ENV_FILE" exec -T "$1" python -c \
    "import pickle;print(pickle.load(open('$2','rb'))['metadata']['version'])" \
    2>/dev/null | tr -d '\r' || echo "unknown"
}

# ---------------------------------------------------------------------------
# UniAD 用 info(motion anchor 込み)
#   出力: $DATA_ROOT/infos/uniad/  →  container: /workspace/UniAD/data/infos
# ---------------------------------------------------------------------------
prepare_uniad() {
  local out="$DATA_ROOT/infos/uniad" others="$DATA_ROOT/others"
  mkdir -p "$out" "$others"

  if [ -f "$others/motion_anchor_infos_mode6.pkl" ]; then
    skip "uniad motion anchor present"
  else
    log "downloading motion anchor"
    wget -q -O "$others/motion_anchor_infos_mode6.pkl" "$ANCHOR_URL"
  fi

  if [ -f "$out/nuscenes_infos_temporal_val.pkl" ] && \
     [ "$(_pkl_version uniad2 data/infos/nuscenes_infos_temporal_val.pkl)" = "$VERSION" ]; then
    skip "uniad info ($VERSION) present"
    return
  fi
  log "generating UniAD info ($VERSION)"
  $COMPOSE --env-file "$ENV_FILE" up -d uniad2
  $COMPOSE --env-file "$ENV_FILE" exec -T uniad2 bash -c \
    "cd /workspace/UniAD && python tools/create_data.py nuscenes \
       --root-path ./data/nuscenes --out-dir ./data/infos \
       --extra-tag nuscenes --version $VERSION --canbus ./data/nuscenes"
  $COMPOSE --env-file "$ENV_FILE" exec -T uniad2 python -c \
    "import pickle;d=pickle.load(open('data/infos/nuscenes_infos_temporal_val.pkl','rb'));\
print('uniad:',d['metadata']['version'],len(d['infos']))"
}

# ---------------------------------------------------------------------------
# VAD 用 info(vad_ 接頭辞。VAD 独自コンバータで生成)
#   出力: $DATA_ROOT/infos/vad/  →  container: /workspace/VAD/data/nuscenes 直下
#   NOTE: VAD は pkl を data/nuscenes 直下に置く流儀。compose のマウントに合わせる。
# ---------------------------------------------------------------------------
prepare_vad() {
  local out="$DATA_ROOT/infos/vad"
  mkdir -p "$out"

  if [ -f "$out/vad_nuscenes_infos_temporal_val.pkl" ] && \
     [ "$(_pkl_version vad data/infos/vad_nuscenes_infos_temporal_val.pkl)" = "$VERSION" ]; then
    skip "vad info ($VERSION) present"
    return
  fi
  log "generating VAD info ($VERSION)"
  $COMPOSE --env-file "$ENV_FILE" up -d vad
  # 引数名は VAD の版で揺れるため docs/prepare_dataset.md で要確認
  $COMPOSE --env-file "$ENV_FILE" exec -T vad bash -c \
    "cd /workspace/VAD && python tools/data_converter/vad_nuscenes_converter.py nuscenes \
       --root-path ./data/nuscenes --out-dir ./data/infos \
       --extra-tag vad_nuscenes --version $VERSION --canbus ./data/nuscenes"
  $COMPOSE --env-file "$ENV_FILE" exec -T vad python -c \
    "import pickle;d=pickle.load(open('data/infos/vad_nuscenes_infos_temporal_val.pkl','rb'));\
print('vad:',d['metadata']['version'],len(d['infos']))"
}

# ---------------------------------------------------------------------------
# SparseDrive 用 info(独自 info + kmeans anchor 前処理)
#   出力: $DATA_ROOT/infos/sparsedrive/
#   NOTE: SparseDrive は create_data + anchor 生成の2段。スクリプト名は要確認。
# ---------------------------------------------------------------------------
prepare_sparsedrive() {
  local out="$DATA_ROOT/infos/sparsedrive"
  mkdir -p "$out"

  if [ -f "$out/nuscenes_infos_val.pkl" ] || [ -f "$out/sparsedrive_nuscenes_infos_val.pkl" ]; then
    skip "sparsedrive info present (存在チェックのみ。作り直すなら手動削除)"
    return
  fi
  log "generating SparseDrive info + anchors ($VERSION)"
  $COMPOSE --env-file "$ENV_FILE" up -d sparsedrive
  # SparseDrive 公式手順に相当(リポジトリ同梱の create_data / kmeans)。
  # 実スクリプト名は SparseDrive/tools 配下を確認して合わせること。
  $COMPOSE --env-file "$ENV_FILE" exec -T sparsedrive bash -c \
    "cd /workspace/SparseDrive && \
     python tools/data_converter/nuscenes_converter.py nuscenes \
       --root-path ./data/nuscenes --out-dir ./data/infos \
       --extra-tag nuscenes --version $VERSION --canbus ./data/nuscenes && \
     ( [ -f tools/kmeans/kmeans_map.py ] && python tools/kmeans/kmeans_map.py || true ) && \
     ( [ -f tools/kmeans/kmeans_motion.py ] && python tools/kmeans/kmeans_motion.py || true )"
  log "sparsedrive info generated (件数は SparseDrive 側ログを確認)"
}

# ---------------------------------------------------------------------------
main() {
  local target="${1:-all}"
  case "$target" in
    raw)          prepare_raw ;;
    uniad)        prepare_raw; prepare_uniad ;;
    vad)          prepare_raw; prepare_vad ;;
    sparsedrive)  prepare_raw; prepare_sparsedrive ;;
    all)          prepare_raw; prepare_uniad; prepare_vad; prepare_sparsedrive ;;
    *)            err "unknown target: $target (raw|uniad|vad|sparsedrive|all)" ;;
  esac
  log "done: $target"
}
main "$@"
