# UniAD v2.0 + WorldEngine — Unified Setup Guide (conda-free)
## End-to-End Pre-training × World Model × RL Post-training

> **Prerequisites**: Python 3.9 and CUDA 11.8 must be available at the OS level.  
> No conda is used anywhere. Environments are managed with Python's built-in `venv`.  
> If you are using Docker, the accompanying `Dockerfile.{uniad2,algengine,simengine}` files map directly to each environment described here.  
> **Last updated**: 2026-04-26

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Hardware Requirements](#2-hardware-requirements)
3. [OS-Level Prerequisites](#3-os-level-prerequisites)
4. [Creating the Three Virtual Environments](#4-creating-the-three-virtual-environments)
5. [uniad2 Environment — UniAD v2.0 Pre-training](#5-uniad2-environment)
6. [algengine Environment — AlgEngine RL Fine-tuning](#6-algengine-environment)
7. [simengine Environment — SimEngine 3DGS Closed-Loop](#7-simengine-environment)
8. [Persisting Environment Variables](#8-persisting-environment-variables)
9. [Dataset Preparation](#9-dataset-preparation)
10. [Integrating Rare-Condition Datasets](#10-integrating-rare-condition-datasets)
11. [Data Augmentation Implementation](#11-data-augmentation-implementation)
12. [Two-Stage Training Pipeline (UniAD)](#12-two-stage-training-pipeline)
13. [WorldEngine Post-training Pipeline](#13-worldengine-post-training-pipeline)
14. [Evaluation and Benchmarks](#14-evaluation-and-benchmarks)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Architecture Overview

```
Camera inputs (6–8 views)
        │
        ▼
┌─────────────────┐
│   BEVFormer     │  ← ResNet-101-DCN + Deformable Attention
│  (BEV encoder)  │    Temporal BEV aggregation (queue_length=3–5)
└────────┬────────┘
         │ BEV embed
    ┌────┴──────────────────────────────┐
    ▼        ▼          ▼         ▼
┌──────┐ ┌──────┐  ┌──────┐  ┌──────┐
│Track │ │ Map  │  │Motion│  │ Occ  │
│Head  │ │Head  │  │Head  │  │Head  │
└──┬───┘ └──┬───┘  └──┬───┘  └──┬───┘
   └─────────┴──────────┴────────┘
                   │
             ┌─────▼──────┐
             │  Planning  │  ← Primary target of WorldEngine RL
             │   Head     │
             └────────────┘
                   │
       ┌───────────▼──────────────┐
       │      WorldEngine         │
       │  SimEngine  +  AlgEngine │  ← Post-training layer
       └──────────────────────────┘
```

UniAD integrates five perception and prediction tasks — tracking, mapping, motion forecasting, occupancy prediction, and trajectory planning — into a single end-to-end model built on top of BEVFormer. WorldEngine then applies RL-based post-training focused on rare, safety-critical scenarios discovered automatically from real driving logs.

---

## 2. Hardware Requirements

| Phase | Minimum | Recommended |
|-------|---------|-------------|
| Stage 1 Perception pre-training | 8 × A100 (80 GB) | 16 × A100 |
| Stage 2 E2E pre-training | 8 × A100 (80 GB) | 16 × A100 |
| SimEngine rollout generation | 4 × RTX 3090 | 4 × A100 |
| RL fine-tuning | 8 × A100 | 8 × A100 |
| Inference / evaluation | 1 × RTX 3090 | 1 × A100 |
| Storage | 1 TB NVMe | 4 TB NVMe |

> `queue_length` (number of BEV frames aggregated temporally) has a large impact on VRAM usage. Reduce it to 3 if you hit memory limits.

---

## 3. OS-Level Prerequisites

The steps below assume **Ubuntu 22.04 LTS**. Verify each dependency before proceeding.

```bash
# Confirm OS
lsb_release -a

# Confirm CUDA driver (525+ required for CUDA 11.8)
nvidia-smi
# Expected: Driver Version 525.xx.xx  |  CUDA Version: 11.8

# Confirm Python 3.9 is already installed
python3.9 --version   # Python 3.9.x

# Build tools and system libraries
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git git-lfs ninja-build \
    libgl1-mesa-glx libglib2.0-0 libgeos-dev \
    libgl1-mesa-dri libegl1-mesa-dev libgles2-mesa-dev \
    ffmpeg gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60
```

**If Python 3.9 is not yet installed** (deadsnakes PPA):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils python3.9-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
```

Unlike conda, `venv` does not bundle a Python interpreter. Python 3.9 must exist at the OS level before any venv is created.

---

## 4. Creating the Three Virtual Environments

All three environments are created up front and activated on demand with `source`.

```bash
mkdir -p ~/venvs

# (1) UniAD v2.0 pre-training
python3.9 -m venv ~/venvs/uniad2

# (2) WorldEngine AlgEngine — E2E training and RL fine-tuning
python3.9 -m venv ~/venvs/algengine

# (3) WorldEngine SimEngine — 3DGS closed-loop simulation
python3.9 -m venv ~/venvs/simengine
```

Switch environments with:

```bash
source ~/venvs/uniad2/bin/activate      # UniAD pre-training
source ~/venvs/algengine/bin/activate   # AlgEngine RL
source ~/venvs/simengine/bin/activate   # SimEngine 3DGS
```

Each section below begins with the required `source` line.

---

## 5. uniad2 Environment

Used for UniAD v2.0 Stage 1 and Stage 2 pre-training. This environment can use OpenMMLab's pre-built `mmcv-full` wheel, which makes setup straightforward.

```bash
source ~/venvs/uniad2/bin/activate
pip install --upgrade pip setuptools wheel
```

### 5.1 PyTorch 2.0.1 (CUDA 11.8)

```bash
pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 5.2 mmcv-full (pre-built wheel)

The `uniad2` environment uses the hosted binary wheel — no source compilation needed.

```bash
pip install mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### 5.3 OpenMMLab Ecosystem

```bash
pip install \
    mmdet==2.26.0 \
    mmsegmentation==0.29.1 \
    mmdet3d==1.0.0rc6
```

### 5.4 Clone UniAD and Install Dependencies

```bash
git clone -b v2.0 https://github.com/OpenDriveLab/UniAD.git ~/UniAD
cd ~/UniAD
pip install -r requirements.txt
```

### 5.5 nuScenes Devkit and Additional Packages

```bash
pip install \
    nuscenes-devkit==1.1.11 \
    shapely==1.8.5 \
    pyquaternion \
    motmetrics \
    networkx \
    scikit-image \
    descartes \
    albumentations==1.3.1 \
    opencv-python-headless==4.8.1.78 \
    kornia==0.7.0
```

### 5.6 Download Pre-trained Weights

```bash
mkdir -p ~/UniAD/ckpts && cd ~/UniAD/ckpts

HF_BASE="https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts"
wget -q $HF_BASE/r101_dcn_fcos3d_pretrain.pth
wget -q $HF_BASE/bevformer_r101_dcn_24ep.pth
wget -q $HF_BASE/uniad_base_track_map.pth
wget -q $HF_BASE/uniad_base_e2e.pth
```

### 5.7 Verify Installation

```bash
python - <<'EOF'
import torch, mmdet3d
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
print(f"mmdet3d : {mmdet3d.__version__}")
from mmdet3d.models import build_detector
print("mmdet3d import OK")
EOF
```

---

## 6. algengine Environment

Used for WorldEngine AlgEngine: end-to-end model training, open-loop evaluation, and RL fine-tuning. **MMCV must be compiled from source** here because AlgEngine requires custom CUDA operators not shipped in the pre-built wheel.

```bash
source ~/venvs/algengine/bin/activate
pip install --upgrade pip
# Pin setuptools — newer versions break the MMCV build
pip install setuptools==75.1.0 wheel
```

### 6.1 PyTorch 2.0.1

```bash
pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 6.2 MMCV Source Build (10–15 minutes)

```bash
git clone https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv && git checkout v1.6.2

MMCV_WITH_OPS=1 pip install -v -e .

# Verify the build
python .dev_scripts/check_installation.py
```

### 6.3 OpenMMLab Ecosystem

```bash
pip install \
    mmcls==0.25.0 \
    mmdet==2.25.3 \
    mmdet3d==1.0.0rc6 \
    mmsegmentation==0.29.1
```

### 6.4 Clone WorldEngine and NAVSIM

```bash
git clone https://github.com/OpenDriveLab/WorldEngine.git ~/WorldEngine

# NAVSIM devkit v1.1 is required for evaluation metrics
git clone -b v1.1 https://github.com/autonomousvision/navsim.git ~/navsim
```

### 6.5 AlgEngine-Specific Dependencies

```bash
cd ~/WorldEngine/projects/AlgEngine
pip install -r requirements.txt
pip install shapely==2.0.4
```

### 6.6 RL and Distributed Training Tools

```bash
pip install \
    ray==2.9.3 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    tensorboard==2.14.0 \
    pandas \
    scikit-learn \
    scipy \
    tqdm
```

### 6.7 Verify Installation

```bash
python - <<'EOF'
import torch, mmcv, mmdet3d
print(f"PyTorch : {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"mmcv    : {mmcv.__version__}")
print(f"mmdet3d : {mmdet3d.__version__}")
EOF
```

---

## 7. simengine Environment

Used for WorldEngine SimEngine: photorealistic 3DGS closed-loop simulation and rollout generation. `gsplat` **must** be built with `--no-build-isolation` so it links against the already-installed PyTorch in this venv.

```bash
source ~/venvs/simengine/bin/activate
pip install --upgrade pip
pip install setuptools==75.1.0 wheel
```

### 7.1 PyTorch 2.0.1

```bash
pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 7.2 gsplat v1.4.0 (Gaussian Splatting Core Library)

`--no-build-isolation` is required so the build finds the venv's PyTorch installation rather than fetching a fresh one.

```bash
pip install ninja
pip install \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" \
    --no-build-isolation
```

### 7.3 Ray and Hydra

```bash
pip install \
    "ray[default]==2.9.3" \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    aiohttp aiohttp-cors prometheus_client
```

### 7.4 Clone SimEngine and Install Dependencies

```bash
# Clone WorldEngine if not already present
[ -d ~/WorldEngine ] || \
    git clone https://github.com/OpenDriveLab/WorldEngine.git ~/WorldEngine

# MTGS — Multi-Traversal Gaussian Splatting scene reconstruction
git clone https://github.com/OpenDriveLab/MTGS.git ~/MTGS

# NAVSIM devkit
[ -d ~/navsim ] || \
    git clone -b v1.1 https://github.com/autonomousvision/navsim.git ~/navsim

cd ~/WorldEngine/projects/SimEngine
pip install -r requirements.txt

pip install \
    nuscenes-devkit==1.1.11 \
    numpy==1.24.4 \
    scipy pandas tqdm pillow \
    opencv-python-headless==4.8.1.78 \
    pyquaternion shapely==2.0.4
```

### 7.5 Headless EGL Rendering

Required for 3DGS rendering on a GPU server without a display.

```bash
# Add to ~/.bashrc (or set per-session before running SimEngine)
echo 'export PYOPENGL_PLATFORM=egl' >> ~/.bashrc
echo 'export EGL_DEVICE_ID=0'       >> ~/.bashrc
```

### 7.6 Verify Installation

```bash
python - <<'EOF'
import torch, gsplat, ray
print(f"PyTorch : {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"gsplat  : {gsplat.__version__}")
print(f"ray     : {ray.__version__}")
EOF
```

---

## 8. Persisting Environment Variables

Add the following to `~/.bashrc` and run `source ~/.bashrc`.

```bash
# ── Common ────────────────────────────────────────────────
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
```

Because `venv` does not manage `PYTHONPATH`, define shell functions that set it automatically on activation:

```bash
# Add to ~/.bashrc

act-uniad2() {
    source ~/venvs/uniad2/bin/activate
    export PYTHONPATH=$UNIAD_ROOT:$UNIAD_ROOT/projects:$PYTHONPATH
    echo "[uniad2] activated"
}

act-alge() {
    source ~/venvs/algengine/bin/activate
    export PYTHONPATH=$WORLDENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
    echo "[algengine] activated"
}

act-sime() {
    source ~/venvs/simengine/bin/activate
    export PYTHONPATH=$WORLDENGINE_ROOT:$SIMENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:~/MTGS:$PYTHONPATH
    export PYOPENGL_PLATFORM=egl
    export GSPLAT_BACKEND=cuda
    echo "[simengine] activated"
}
```

The rest of this guide uses `act-uniad2`, `act-alge`, and `act-sime` as shorthand.

---

## 9. Dataset Preparation

### 9.1 Overall Directory Layout

```
~/
├── UniAD/
│   ├── ckpts/
│   └── data/
│       ├── nuscenes/          ← full nuScenes v1.0-trainval
│       ├── nuscenes_c/        ← corruption variants (generated)
│       ├── acdc_nuscenes/     ← ACDC converted to nuScenes format
│       ├── infos/
│       │   ├── nuscenes_infos_temporal_train.pkl
│       │   └── nuscenes_infos_temporal_val.pkl
│       └── others/
│           └── motion_anchor_infos_mode6.pkl
└── WorldEngine/
    └── data/
        ├── raw/
        │   ├── nuscenes/      ← symlink or copy
        │   └── nuplan/
        │       ├── maps/
        │       └── openscene/
        └── alg_engine/
            ├── ckpts/
            ├── merged_infos_navformer/
            │   ├── nuplan_openscene_navtrain.pkl
            │   └── nuplan_openscene_navtest.pkl
            └── openscene-synthetic/   ← SimEngine rollout output
```

### 9.2 nuScenes Info Files for UniAD

**Option A — Download from HuggingFace (recommended)**

```bash
act-uniad2
cd ~/UniAD
mkdir -p data/infos data/others

HF="https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data"
wget -q -P data/infos $HF/nuscenes_infos_temporal_train.pkl
wget -q -P data/infos $HF/nuscenes_infos_temporal_val.pkl
wget -q -P data/others $HF/motion_anchor_infos_mode6.pkl
```

**Option B — Generate locally**

```bash
cd ~/UniAD/data && mkdir infos
bash ../tools/uniad_create_data.sh
# Note: set data_root='' in config afterwards (see Issue #13)
```

### 9.3 WorldEngine Official Dataset (HuggingFace)

```bash
act-alge
pip install huggingface_hub

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='OpenDriveLab/WorldEngine',
    repo_type='dataset',
    local_dir=os.path.join(os.environ['WORLDENGINE_ROOT'], 'data'),
)
EOF
```

---

## 10. Integrating Rare-Condition Datasets

nuScenes covers predominantly clear-weather driving. The following datasets supplement rare and adverse scenarios.

| Dataset | Key characteristics | License |
|---------|---------------------|---------|
| **ACDC** | Real-world fog / rain / night / snow | Free for research |
| **nuScenes-C** | 16 corruption types applied to nuScenes | Apache 2.0 |
| **Adver-City** | CARLA-synthesized, 6 weather conditions | Open |
| **Bench2Drive** | CARLA 44 interactive scenarios, 2 M frames | Apache 2.0 |

### 10.1 ACDC Conversion Script

Place this at `~/UniAD/tools/data_converter/acdc_to_nuscenes.py`:

```python
"""Convert ACDC dataset to nuScenes-compatible format (skeleton implementation)."""
import pickle
from pathlib import Path
from typing import Dict, List

CONDITION_MAP = {
    'fog': 'adverse_fog', 'night': 'adverse_night',
    'rain': 'adverse_rain', 'snow': 'adverse_snow',
}

def convert_acdc_scene(scene_dir: Path, condition: str) -> Dict:
    scene_token = f"acdc_{condition}_{scene_dir.name}"
    samples = []
    for img_file in sorted(scene_dir.glob('*.png')):
        samples.append({
            'token': f"{scene_token}_{img_file.stem}",
            'timestamp': int(img_file.stem),
            'scene_token': scene_token,
            'data': {'CAM_FRONT': {
                'filename': str(img_file),
                'sensor_modality': 'camera',
                'channel': 'CAM_FRONT',
            }},
            'condition': CONDITION_MAP.get(condition, condition),
        })
    return {'scene_token': scene_token, 'samples': samples}

def main(acdc_root: str, out_dir: str, conditions: List[str]):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_infos = []
    for cond in conditions:
        cond_dir = Path(acdc_root) / 'rgb_anon' / cond / 'val'
        if not cond_dir.exists():
            print(f"[WARN] {cond_dir} not found, skipping.")
            continue
        for scene_dir in sorted(cond_dir.iterdir()):
            if scene_dir.is_dir():
                all_infos.append(convert_acdc_scene(scene_dir, cond))
    out_pkl = out_path / 'acdc_infos_val.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_infos, f)
    print(f"Saved {len(all_infos)} scenes → {out_pkl}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--acdc-root', required=True)
    p.add_argument('--out-dir',   required=True)
    p.add_argument('--conditions', nargs='+',
                   default=['fog', 'night', 'rain', 'snow'])
    args = p.parse_args()
    main(args.acdc_root, args.out_dir, args.conditions)
```

```bash
act-uniad2
python ~/UniAD/tools/data_converter/acdc_to_nuscenes.py \
    --acdc-root /path/to/acdc \
    --out-dir ~/UniAD/data/acdc_nuscenes
```

### 10.2 Generating nuScenes-C Corruption Variants

```bash
act-uniad2
pip install nuscenes-c-tools

python -m nuscenes_c.generate \
    --data-root ~/UniAD/data/nuscenes \
    --out-dir   ~/UniAD/data/nuscenes_c \
    --corruptions fog rain night motion_blur \
    --severity 1 2 3   # 1 = mild, 3 = severe
```

Generated corruption types: `fog`, `rain`, `snow` (weather); `night`, `low_light`, `overexposure` (lighting); `motion_blur`, `camera_crash`, `frame_lost` (sensor); `gaussian_noise`, `impulse_noise`, `color_quant` (spatial).

---

## 11. Data Augmentation Implementation

Place this file at `~/UniAD/projects/mmdet3d_plugin/uniad/utils/weather_augment.py`:

```python
"""
Consistent multi-view weather augmentation Transform for UniAD.
Applies the same weather condition to all camera views simultaneously
to preserve cross-view consistency.
"""
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RandomWeatherAugmentation:
    """
    Apply a randomly chosen weather effect uniformly across all camera views.

    Args:
        fog_prob   (float): Probability of applying fog.
        rain_prob  (float): Probability of applying rain streaks.
        night_prob (float): Probability of applying night-time darkening.
        blur_prob  (float): Probability of applying motion blur.
        snow_prob  (float): Probability of applying snow particles.
    """

    def __init__(self, fog_prob=0.15, rain_prob=0.15,
                 night_prob=0.10, blur_prob=0.10, snow_prob=0.05):
        self.fog_prob   = fog_prob
        self.rain_prob  = rain_prob
        self.night_prob = night_prob
        self.blur_prob  = blur_prob
        self.snow_prob  = snow_prob

    def _add_fog(self, img, severity=0.5):
        """Koschmieder atmospheric model fog simulation."""
        h, w = img.shape[:2]
        depth = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        t = np.exp(-severity * depth * 3.0)
        fog_color = np.array([210, 210, 210], dtype=np.float32)
        return np.clip(
            img.astype(np.float32) * t[..., None]
            + fog_color * (1 - t[..., None]), 0, 255
        ).astype(np.uint8)

    def _add_rain(self, img, intensity=800):
        """Overlay random rain streak lines."""
        rain = img.copy()
        h, w = img.shape[:2]
        for _ in range(intensity):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            length  = np.random.randint(10, 30)
            angle   = np.random.uniform(-20, 20)
            x2 = int(x1 + length * np.sin(np.radians(angle)))
            y2 = int(y1 + length * np.cos(np.radians(angle)))
            cv2.line(rain, (x1, y1), (x2, y2), (200, 200, 200), 1)
        return cv2.addWeighted(img, 0.7, rain, 0.3, 0)

    def _add_night(self, img):
        """Gamma correction to simulate low-light nighttime conditions."""
        gamma = np.random.uniform(2.5, 4.0)
        lut = np.array(
            [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(img, lut)

    def _add_motion_blur(self, img):
        """Horizontal motion blur kernel."""
        k = np.random.choice([7, 11, 15])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = np.ones(k) / k
        return cv2.filter2D(img, -1, kernel)

    def _add_snow(self, img, intensity=500):
        """Scatter small white circles to simulate snowflakes."""
        snow = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        for _ in range(intensity):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.circle(snow, (x, y), np.random.randint(1, 3),
                       (255, 255, 255), -1)
        return np.clip(snow, 0, 255).astype(np.uint8)

    def __call__(self, results: dict) -> dict:
        r = np.random.random()
        aug_func = None
        cumprob = 0.0
        for prob, func in [
            (self.fog_prob,   self._add_fog),
            (self.rain_prob,  self._add_rain),
            (self.night_prob, self._add_night),
            (self.blur_prob,  self._add_motion_blur),
            (self.snow_prob,  self._add_snow),
        ]:
            cumprob += prob
            if r < cumprob:
                aug_func = func
                break
        if aug_func is None:
            return results  # no augmentation this sample
        # Apply the same effect to every camera view
        results['img'] = [aug_func(img) for img in results.get('img', [])]
        return results
```

Register this transform in your Stage 2 config:

```python
# projects/configs/stage2_e2e/base_e2e_adverse.py
_base_ = ['./base_e2e.py']

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', ...),
    dict(
        type='RandomWeatherAugmentation',
        fog_prob=0.15, rain_prob=0.15,
        night_prob=0.10, blur_prob=0.10, snow_prob=0.05,
    ),
    dict(type='NormalizeMultiviewImage', ...),
    dict(type='PadMultiViewImage', ...),
    dict(type='DefaultFormatBundle3D', ...),
    dict(type='Collect3D', ...),
]
```

---

## 12. Two-Stage Training Pipeline (UniAD)

```bash
act-uniad2
cd ~/UniAD
```

### 12.1 Stage 1 — Perception Pre-training (20 epochs)

Trains: BEVFormer backbone, TrackHead, MapHead.

```bash
./tools/dist_train.sh \
    projects/configs/stage1_track_map/base_track_map.py \
    8 \
    --work-dir work_dirs/stage1_base
```

### 12.2 Stage 2 — End-to-End Training (6 epochs)

Trains all heads jointly: TrackHead, MapHead, MotionHead, OccHead, PlanningHead.

```bash
# Standard config
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e.py \
    8 \
    --work-dir work_dirs/stage2_e2e

# With adverse-weather augmentation
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e_adverse.py \
    8 \
    --work-dir work_dirs/stage2_e2e_adverse
```

### 12.3 Training Schedule

```python
# Optimizer and LR schedule (inside the config file)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)}
    )
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type='EpochBasedRunner', max_epochs=20)  # Stage 1
# Stage 2: max_epochs=6
```

### 12.4 Resuming and Quick Sanity Check

```bash
# Resume from a checkpoint
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e.py 8 \
    --resume-from work_dirs/stage2_e2e/epoch_3.pth

# Quick check on nuScenes-mini (single GPU, 2 epochs)
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e.py 1 \
    --cfg-options \
        data.train.ann_file='data/infos/nuscenes_infos_temporal_train_mini.pkl' \
        data.val.ann_file='data/infos/nuscenes_infos_temporal_val_mini.pkl' \
        runner.max_epochs=2 \
        evaluation.interval=1
```

---

## 13. WorldEngine Post-training Pipeline

WorldEngine identifies failure scenarios from real driving logs, reconstructs them photorealistically with 3DGS, and uses RL to fine-tune the planner on those hard cases. The full pipeline yields a **+15.28 pp absolute improvement** in closed-loop success rate (73.61% → 88.89%).

### 13.1 Phase 2-A — Open-loop Evaluation and Rare Case Extraction

```bash
act-alge
cd $ALGENGINE_ROOT

# Open-loop evaluation on the full navtest split
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_uniad_50pct.py \
    work_dirs/e2e_uniad_50pct/epoch_20.pth \
    8
# Output: work_dirs/e2e_uniad_50pct/navtest.csv

# Extract three categories of rare/failure scenarios
python scripts/rare_case_sampling_by_pdms.py \
    --pdm-result work_dirs/e2e_uniad_50pct/navtest.csv \
    --base-split configs/navsim_splits/navtest_split/navtest.yaml \
    --output-dir configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare
```

Extracted splits:

```
configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare/
├── navtest_collision.yaml    ← no_at_fault_collisions < 1.0
├── navtest_off_road.yaml     ← drivable_area_compliance < 1.0
└── navtest_ep_1pct.yaml      ← ego_progress bottom 1%
```

### 13.2 Phase 2-B — SimEngine Rollout Generation

```bash
act-sime
cd $SIMENGINE_ROOT

# Quick sanity check: run on 288 pre-identified rare test scenarios
bash scripts/closed_loop_test.sh

# Distributed rollout generation from rare training scenarios
bash scripts/run_ray_distributed_rollout.sh \
    $ALGENGINE_ROOT/configs/worldengine/e2e_uniad_50pct.py \
    $ALGENGINE_ROOT/work_dirs/e2e_uniad_50pct/epoch_20.pth \
    e2e_uniad_50pct \
    navtrain_50pct_collision \
    navtrain

# Convert rollout output to AlgEngine training format
python scripts/export_simulation_data.py \
    --test_path experiments/closed_loop_exps/e2e_uniad_50pct/navtrain_NR \
    --appendix $(date +%Y%m%d)
# Output: data/alg_engine/openscene-synthetic/
```

**Reactive vs Non-Reactive modes:**

| Mode | Flag | Description | Use case |
|------|------|-------------|----------|
| Non-Reactive | `NR` | Other agents follow fixed trajectories | Fast initial evaluation |
| Reactive | `R` | Other agents respond with IDM/CBV | Realistic closed-loop RL training |

### 13.3 Phase 2-C — RL Fine-tuning

```bash
act-alge
cd $ALGENGINE_ROOT

./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    8 \
    work_dirs/e2e_uniad_50pct/epoch_20.pth
```

Key differences in the RL fine-tune config:

```python
# configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py
_base_ = ['./e2e_uniad_50pct.py']

data = dict(
    train=dict(
        # Filter to rare failure scenarios only
        scenario_filter=[
            'configs/navsim_splits/navtrain_split/e2e_uniad_50pct_ep20/'
            'navtrain_50pct_collision.yaml',
            'configs/navsim_splits/navtrain_split/e2e_uniad_50pct_ep20/'
            'navtrain_50pct_off_road.yaml',
        ],
        # Include synthetic rollout data from SimEngine
        extra_ann_files=[
            'data/alg_engine/openscene-synthetic/meta_datas/',
        ],
    )
)

# Lower LR than pre-training to avoid catastrophic forgetting
optimizer = dict(type='AdamW', lr=5e-5)   # was 2e-4 in pre-training
runner    = dict(type='EpochBasedRunner', max_epochs=8)
```

---

## 14. Evaluation and Benchmarks

### 14.1 Standard UniAD Evaluation (nuScenes val)

```bash
act-uniad2
cd ~/UniAD

./tools/dist_test.sh \
    projects/configs/stage2_e2e/base_e2e.py \
    ckpts/uniad_base_e2e.pth \
    8 --eval bbox

# Expected results (UniAD-B, R101):
# Tracking  AMOTA  : 0.380
# Mapping   IoU    : 0.314
# Motion    minADE : 0.794
# Occupancy IoU-n  : 64.0
# Planning  Col.   : 0.29%
```

### 14.2 Adverse-Weather Robustness Evaluation (nuScenes-C)

```bash
act-uniad2
cd ~/UniAD

for corruption in fog rain night motion_blur; do
    for severity in 1 2 3; do
        ./tools/dist_test.sh \
            projects/configs/stage2_e2e/base_e2e.py \
            ckpts/uniad_base_e2e.pth 4 --eval bbox \
            --cfg-options \
            data.test.ann_file="data/nuscenes_c/${corruption}_s${severity}_infos_val.pkl"
    done
done
```

### 14.3 WorldEngine Closed-Loop Evaluation

```bash
act-sime
cd $ALGENGINE_ROOT

bash scripts/run_ray_distributed_testing.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
    e2e_uniad_rlft \
    navtest_failures \
    NR

# Expected results after the full WorldEngine pipeline:
# Closed-loop Success Rate : 88.89%  (+15.28 pp vs base)
# Closed-loop PDMS*        : 70.12   (+9.84  vs base)
```

### 14.4 Bench2Drive Closed-Loop Evaluation (optional)

```bash
act-alge
git clone https://github.com/Thinklab-SJTU/Bench2Drive ~/Bench2Drive
cd ~/Bench2Drive

python eval/eval_b2d.py \
    --agent UniADAgent \
    --checkpoint $ALGENGINE_ROOT/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
    --routes routes/bench2drive220.xml
```

---

## 15. Troubleshooting

### `AssertionError: data_root is not empty`

Locally generated pkl files embed absolute paths. Fix it in the config:

```python
data = dict(train=dict(data_root='', ...))
```

### `CUDA out of memory`

Reduce the number of BEV frames aggregated:

```python
model = dict(
    img_bev_encoder_backbone=dict(queue_length=3)  # default 5 → 3
)
```

Or enable gradient checkpointing in the backbone:

```python
model = dict(img_backbone=dict(with_cp=True))
```

### mmcv version mismatch

```bash
# uniad2 — reinstall the pre-built wheel
source ~/venvs/uniad2/bin/activate
pip install mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    --force-reinstall

# algengine — rebuild from source
source ~/venvs/algengine/bin/activate
cd ~/mmcv
MMCV_WITH_OPS=1 pip install -v -e . --force-reinstall
```

### gsplat build failure

```bash
# Confirm CUDA headers are present
ls /usr/local/cuda/include/cuda_runtime.h

# --no-build-isolation is mandatory
source ~/venvs/simengine/bin/activate
pip install "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" \
    --no-build-isolation --force-reinstall
```

### Stage 1 AMOTA lower than reported

```python
# Verify in base_track_map.py:
loss_past_traj = False   # must be False
# Confirm img_neck and BN parameters are NOT frozen
```

### Ray distributed simulation hangs

```bash
ray stop --force
pkill -f "ray::"
# Restart
source ~/venvs/simengine/bin/activate
ray start --head --num-gpus=8
```

### `PYTHONPATH` missing after activation

If you sourced the venv directly instead of calling the shell functions defined in §8, re-export manually:

```bash
# For algengine
export PYTHONPATH=$WORLDENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
```

---

## Appendix: Recommended Directory Layout

```
~/
├── venvs/
│   ├── uniad2/        ← source ~/venvs/uniad2/bin/activate
│   ├── algengine/     ← source ~/venvs/algengine/bin/activate
│   └── simengine/     ← source ~/venvs/simengine/bin/activate
│
├── UniAD/
│   ├── ckpts/
│   ├── data/
│   │   ├── nuscenes/
│   │   ├── nuscenes_c/
│   │   ├── acdc_nuscenes/
│   │   ├── infos/
│   │   └── others/
│   ├── projects/
│   │   ├── configs/
│   │   │   ├── stage1_track_map/
│   │   │   └── stage2_e2e/
│   │   │       ├── base_e2e.py
│   │   │       └── base_e2e_adverse.py
│   │   └── mmdet3d_plugin/uniad/
│   │       ├── datasets/mixed_adverse_dataset.py
│   │       └── utils/weather_augment.py
│   └── work_dirs/
│
├── WorldEngine/
│   ├── data/
│   │   ├── raw/nuplan/
│   │   ├── alg_engine/
│   │   │   ├── ckpts/
│   │   │   ├── merged_infos_navformer/
│   │   │   └── openscene-synthetic/
│   │   └── sim_engine/
│   └── projects/
│       ├── AlgEngine/
│       │   ├── configs/worldengine/
│       │   └── work_dirs/
│       └── SimEngine/
│           └── experiments/
│
├── mmcv/              ← algengine source build
├── navsim/            ← v1.1 devkit
└── MTGS/              ← simengine 3DGS reconstruction
```

---

## References

- [UniAD GitHub (v2.0)](https://github.com/OpenDriveLab/UniAD/tree/v2.0)
- [UniAD HuggingFace weights](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes)
- [WorldEngine GitHub](https://github.com/OpenDriveLab/WorldEngine)
- [WorldEngine HuggingFace dataset](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)
- [AlgEngine usage guide](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/algengine_usage.md)
- [SimEngine usage guide](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/simengine_usage.md)
- [MTGS — Multi-Traversal Gaussian Splatting](https://github.com/OpenDriveLab/MTGS)
- [NAVSIM v1.1](https://github.com/autonomousvision/navsim)
- [nuScenes official](https://www.nuscenes.org/download)
- [ACDC dataset](https://acdc.vision.ee.ethz.ch/)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [nuScenes-C (RoboBEV)](https://github.com/Daniel-xsy/RoboBEV)
