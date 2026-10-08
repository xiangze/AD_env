#!/usr/bin/env bash
# SparseDriveV2 の学習済み重み・anchor・ResNet-34 backbone を ckpt/ に配置する。
# コンテナ内でもホストでも実行可:
#   bash tools/download_weights.sh [CKPT_DIR]
# 既定の CKPT_DIR は /workspace/SparseDriveV2/ckpt (コンテナ内)。
# 公式スクリプトが参照するファイル名に合わせて保存する:
#   ckpt/resnet34.bin
#   ckpt/kmeans/{path_1024.npy, velocity_256.npy, trajectory_1024_256.npz}
#   ckpt/sparsedrive_navsimv1.ckpt   (<- sparsedrive_navsimv1_92p2.ckpt, PDMS 92.22)
#   ckpt/sparsedrive_navsimv2.ckpt   (<- sparsedrive_navsimv2_90p3.ckpt, EPDMS 90.38)
set -euo pipefail

CKPT_DIR="${1:-${NAVSIM_DEVKIT_ROOT:-/workspace/SparseDriveV2}/ckpt}"
HF=https://huggingface.co
REPO=wenchaosun/SparseDriveV2
mkdir -p "${CKPT_DIR}/kmeans"

fetch () {  # url dest
    local url="$1" dest="$2"
    if [ -s "${dest}" ]; then
        echo "[skip] ${dest}"
        return
    fi
    echo "[get ] ${dest}"
    # HF_TOKEN があれば付与 (rate limit 回避用・任意)
    if [ -n "${HF_TOKEN:-}" ]; then
        wget -q --show-progress --header="Authorization: Bearer ${HF_TOKEN}" -O "${dest}.part" "${url}"
    else
        wget -q --show-progress -O "${dest}.part" "${url}"
    fi
    mv "${dest}.part" "${dest}"
}

# backbone
fetch "${HF}/timm/resnet34.a1_in1k/resolve/main/pytorch_model.bin" "${CKPT_DIR}/resnet34.bin"

# anchors (path / velocity / trajectory)
for f in path_1024.npy velocity_256.npy trajectory_1024_256.npz; do
    fetch "${HF}/${REPO}/resolve/main/${f}" "${CKPT_DIR}/kmeans/${f}"
done

# pretrained checkpoints (評価スクリプトの CHECKPOINT= 名に合わせる)
fetch "${HF}/${REPO}/resolve/main/sparsedrive_navsimv1_92p2.ckpt" "${CKPT_DIR}/sparsedrive_navsimv1.ckpt"
fetch "${HF}/${REPO}/resolve/main/sparsedrive_navsimv2_90p3.ckpt" "${CKPT_DIR}/sparsedrive_navsimv2.ckpt"

echo
echo "done:"
find "${CKPT_DIR}" -maxdepth 2 -type f -exec ls -lh {} \; | awk '{print $5"\t"$9}'
