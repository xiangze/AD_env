#!/usr/bin/env bash
# SparseDriveV2 container entrypoint
#  - conda env "navsim" を有効化
#  - ホストのソースを /workspace/SparseDriveV2 にマウントした場合など、CUDA ops (.so) が
#    無ければ起動時にビルドする (GPU が見えていればそのアーキでビルドされる)
#  - ckpt / exp ディレクトリを用意
set -e

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate navsim

REPO="${NAVSIM_DEVKIT_ROOT:-/workspace/SparseDriveV2}"
OPS_DIR="${REPO}/navsim/agents/sparsedrive/ops"

if [ -d "${OPS_DIR}" ]; then
    if ! ls "${OPS_DIR}"/deformable_aggregation_ext*.so >/dev/null 2>&1 \
       || ! ls "${OPS_DIR}"/deformable_aggregation_with_depth_ext*.so >/dev/null 2>&1; then
        echo "[entrypoint] CUDA ops not found -> building in ${OPS_DIR}"
        (cd "${OPS_DIR}" && FORCE_CUDA=1 python setup.py develop >/tmp/ops_build.log 2>&1) \
            && echo "[entrypoint] ops build OK" \
            || { echo "[entrypoint] ops build FAILED (see /tmp/ops_build.log)"; tail -30 /tmp/ops_build.log; }
    fi
    # editable install がマウントで消えた場合の保険
    python -c "import navsim" 2>/dev/null || pip install -q -e "${REPO}"
fi

mkdir -p "${REPO}/ckpt/kmeans" "${NAVSIM_EXP_ROOT:-${REPO}/exp}" 2>/dev/null || true

exec "$@"
