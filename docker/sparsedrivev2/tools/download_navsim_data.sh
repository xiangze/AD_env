#!/usr/bin/env bash
# NAVSIM (OpenScene) データをダウンロードし、NAVSIM が要求するレイアウトに整形する。
#
#   bash /workspace/tools/download_navsim_data.sh <split> [<split> ...]
#
#   split:
#     maps      nuPlan maps (必須, ~1GB)
#     mini      OpenScene mini (動作確認用)
#     navtrain  学習用 navtrain サブセット (HF から, 大容量 ~450GB)
#     test      navtest 評価用 (openscene test 全体, ~200GB)
#     navhard   navhard_two_stage (NAVSIM v2 / navhard 評価用)
#     warmup    warmup_two_stage
#
# 出力先は $OPENSCENE_DATA_ROOT (既定 /workspace/dataset, ホストからマウント):
#   dataset/
#   ├── maps/
#   ├── navsim_logs/{mini,test,trainval}/
#   ├── sensor_blobs/{mini,test,trainval}/
#   ├── navhard_two_stage/
#   └── warmup_two_stage/
#
# 実体は SparseDriveV2/download/*.sh (公式) を作業ディレクトリで実行し、結果を移動するだけ。
set -euo pipefail

DATA_ROOT="${OPENSCENE_DATA_ROOT:-/workspace/dataset}"
DL_SCRIPTS="${NAVSIM_DEVKIT_ROOT:-/workspace/SparseDriveV2}/download"
WORK="${DATA_ROOT}/_download_tmp"

[ $# -ge 1 ] || { sed -n '2,25p' "$0"; exit 1; }
mkdir -p "${DATA_ROOT}/navsim_logs" "${DATA_ROOT}/sensor_blobs" "${WORK}"

# $1: 公式スクリプトが作る "<x>_navsim_logs" 等のディレクトリ, $2: 移動先, $3: split 名
# tar の中身が <split>/ サブディレクトリ付きでも無しでも対応する
merge_into () {
    local src="$1" dst_parent="$2" split="$3"
    [ -d "${src}" ] || { echo "!! ${src} not found"; return 1; }
    mkdir -p "${dst_parent}/${split}"
    if [ -d "${src}/${split}" ]; then
        rsync -a --remove-source-files "${src}/${split}/" "${dst_parent}/${split}/"
    else
        rsync -a --remove-source-files "${src}/" "${dst_parent}/${split}/"
    fi
    rm -rf "${src}"
}

run_official () {  # script name
    echo "==> running ${1} in ${WORK}"
    (cd "${WORK}" && bash "${DL_SCRIPTS}/${1}")
}

for split in "$@"; do
    case "${split}" in
        maps)
            if [ -d "${DATA_ROOT}/maps" ]; then echo "[skip] maps"; continue; fi
            run_official download_maps.sh
            mv "${WORK}/maps" "${DATA_ROOT}/maps"
            ;;
        mini)
            run_official download_mini.sh
            merge_into "${WORK}/mini_navsim_logs"   "${DATA_ROOT}/navsim_logs"  mini
            merge_into "${WORK}/mini_sensor_blobs"  "${DATA_ROOT}/sensor_blobs" mini
            ;;
        navtrain)
            run_official download_navtrain_hf.sh
            merge_into "${WORK}/trainval_navsim_logs"  "${DATA_ROOT}/navsim_logs"  trainval
            merge_into "${WORK}/trainval_sensor_blobs" "${DATA_ROOT}/sensor_blobs" trainval
            ;;
        test)
            run_official download_test.sh
            merge_into "${WORK}/test_navsim_logs"  "${DATA_ROOT}/navsim_logs"  test
            merge_into "${WORK}/test_sensor_blobs" "${DATA_ROOT}/sensor_blobs" test
            ;;
        navhard)
            run_official download_navhard_two_stage.sh
            rsync -a --remove-source-files "${WORK}/navhard_two_stage/" "${DATA_ROOT}/navhard_two_stage/"
            rm -rf "${WORK}/navhard_two_stage"
            ;;
        warmup)
            run_official download_warmup_two_stage.sh
            rsync -a --remove-source-files "${WORK}/warmup_two_stage/" "${DATA_ROOT}/warmup_two_stage/"
            rm -rf "${WORK}/warmup_two_stage"
            ;;
        *)
            echo "unknown split: ${split}"; exit 1 ;;
    esac
done

rmdir "${WORK}" 2>/dev/null || echo "note: ${WORK} に残りファイルがあります。確認してください。"
echo "dataset layout:"
find "${DATA_ROOT}" -maxdepth 2 -type d | sort
