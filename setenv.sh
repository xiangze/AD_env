# ── 共通 ──────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ── UniAD ─────────────────────────────────────────────────
export UNIAD_ROOT=~/UniAD

# ── WorldEngine ───────────────────────────────────────────
export WORLDENGINE_ROOT=~/WorldEngine
export NAVSIM_DEVKIT_ROOT=~/navsim
export ALGENGINE_ROOT=~/WorldEngine/projects/AlgEngine
export SIMENGINE_ROOT=~/WorldEngine/projects/SimEngine
export NUPLAN_MAPS_ROOT=~/WorldEngine/data/raw/nuplan/maps

# ── venv 切り替えエイリアス (オプション) ─────────────────
alias act-uniad2='source ~/venvs/uniad2/bin/activate'
alias act-alge='source ~/venvs/algengine/bin/activate'
alias act-sime='source ~/venvs/simengine/bin/activate'


act-uniad2() {
    source ~/venvs/uniad2/bin/activate
    export PYTHONPATH=$UNIAD_ROOT:$UNIAD_ROOT/projects:$PYTHONPATH
}

act-alge() {
    source ~/venvs/algengine/bin/activate
    export PYTHONPATH=$WORLDENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
}

act-sime() {
    source ~/venvs/simengine/bin/activate
    export PYTHONPATH=$WORLDENGINE_ROOT:$SIMENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:~/MTGS:$PYTHONPATH
    export PYOPENGL_PLATFORM=egl
    export GSPLAT_BACKEND=cuda
}