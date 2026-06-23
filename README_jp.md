# UniAD v2.0 + WorldEngine — 統合環境構築ガイド (conda なし)
## End-to-End 事前学習 × 世界モデル × RL ポストトレーニング

> **前提**: Python 3.9 / CUDA 11.8 が OS レベルで用意されていること。  
> conda は一切使用しない。venv または直接システム pip で管理する。  
> Docker で使う場合は付属の `Dockerfile.{uniad2,algengine,simengine}` がそのまま対応する。  
> **最終更新**: 2026-04-26

---

## 目次

1. [アーキテクチャ概要](#1-アーキテクチャ概要)
2. [ハードウェア要件](#2-ハードウェア要件)
3. [OS レベルの前提](#3-os-レベルの前提)
4. [3 つの仮想環境の作成](#4-3-つの仮想環境の作成)
5. [uniad2 環境 — UniAD v2.0 事前学習](#5-uniad2-環境)
6. [algengine 環境 — AlgEngine RL ファインチューニング](#6-algengine-環境)
7. [simengine 環境 — SimEngine 3DGS クローズドループ](#7-simengine-環境)
8. [環境変数の永続化](#8-環境変数の永続化)
9. [データセット準備](#9-データセット準備)
10. [稀少条件データセットの統合](#10-稀少条件データセットの統合)
11. [データオーグメンテーション実装](#11-データオーグメンテーション実装)
12. [2 段階学習パイプライン (UniAD)](#12-2-段階学習パイプライン)
13. [WorldEngine ポストトレーニングパイプライン](#13-worldengine-ポストトレーニングパイプライン)
14. [評価・ベンチマーク](#14-評価ベンチマーク)
15. [トラブルシューティング](#15-トラブルシューティング)

---

## 1. アーキテクチャ概要

```
カメラ入力 (6〜8視点)
    │
    ▼
┌─────────────────┐
│  BEVFormer      │  ← ResNet-101-DCN + Deformable Attention
│  (BEV特徴抽出)   │    時系列 BEV 特徴集約 (queue_length=3〜5)
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
             │  Planning  │  ← WorldEngine RL でここを重点最適化
             │   Head     │
             └────────────┘
                   │
       ┌───────────▼──────────────┐
       │      WorldEngine         │
       │  SimEngine  +  AlgEngine │  ← ポストトレーニング層
       └──────────────────────────┘
```

---

## 2. ハードウェア要件

| フェーズ | 最小構成 | 推奨構成 |
|---------|---------|---------|
| Stage1 Perception | A100 × 8 (80 GB) | A100 × 16 |
| Stage2 E2E | A100 × 8 (80 GB) | A100 × 16 |
| SimEngine ロールアウト | RTX 3090 × 4 | A100 × 4 |
| RL ファインチューニング | A100 × 8 | A100 × 8 |
| 評価 (推論) | RTX 3090 × 1 | A100 × 1 |
| ストレージ | 1 TB NVMe | 4 TB NVMe |

---

## 3. OS レベルの前提

以下が事前に整っていることを確認する。

```bash
# Ubuntu 22.04 LTS 推奨
lsb_release -a

# CUDA 11.8 ドライバ確認
nvidia-smi
# → Driver Version 525+ / CUDA Version 11.8

# Python 3.9 確認 (deadsnakes PPA 等で導入済みであること)
python3.9 --version   # Python 3.9.x

# ビルドツール
sudo apt-get install -y \
    build-essential cmake git git-lfs ninja-build \
    libgl1-mesa-glx libglib2.0-0 libgeos-dev \
    libgl1-mesa-dri libegl1-mesa-dev libgles2-mesa-dev \
    ffmpeg gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60
```

Python 3.9 が入っていない場合:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
```

---

## 4. 3 つの仮想環境の作成

conda の代わりに **Python 標準の venv** を使う。環境ごとにディレクトリを分け、activate スクリプトで切り替える。

```bash
# 各環境のルートディレクトリ
mkdir -p ~/venvs

# ① UniAD 事前学習用
python3.9 -m venv ~/venvs/uniad2

# ② AlgEngine (RL ファインチューニング) 用
python3.9 -m venv ~/venvs/algengine

# ③ SimEngine (3DGS クローズドループ) 用
python3.9 -m venv ~/venvs/simengine
```

以降、各セクションの冒頭で使用する環境を明示する。  
`source ~/venvs/<env>/bin/activate` で切り替える。

---

## 5. uniad2 環境

UniAD v2.0 の Stage1 / Stage2 事前学習に使用する。

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

### 5.2 mmcv-full (ビルド済みホイール)

uniad2 環境は OpenMMLab の配布ホイールをそのまま使える。

```bash
pip install mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### 5.3 OpenMMLab エコシステム

```bash
pip install \
    mmdet==2.26.0 \
    mmsegmentation==0.29.1 \
    mmdet3d==1.0.0rc6
```

### 5.4 UniAD クローン & 依存

```bash
git clone -b v2.0 https://github.com/OpenDriveLab/UniAD.git ~/UniAD
cd ~/UniAD
pip install -r requirements.txt
```

### 5.5 nuScenes devkit & 追加依存

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

### 5.6 事前学習済み重みの取得

```bash
mkdir -p ~/UniAD/ckpts && cd ~/UniAD/ckpts

wget -q https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/r101_dcn_fcos3d_pretrain.pth
wget -q https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/bevformer_r101_dcn_24ep.pth
wget -q https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_track_map.pth
wget -q https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
```

### 5.7 インストール確認

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

## 6. algengine 環境

WorldEngine AlgEngine の E2E 学習・評価・RL ファインチューニングに使用する。  
mmcv は**カスタム CUDA op 込みのソースビルドが必須**。

```bash
source ~/venvs/algengine/bin/activate
pip install --upgrade pip
pip install setuptools==75.1.0 wheel  # 新しすぎると MMCV ビルドが失敗する
```

### 6.1 PyTorch 2.0.1

```bash
pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 6.2 MMCV ソースビルド (10〜15 分)

```bash
git clone https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv && git checkout v1.6.2

MMCV_WITH_OPS=1 pip install -v -e .

# ビルド確認
python .dev_scripts/check_installation.py
```

### 6.3 OpenMMLab エコシステム

```bash
pip install \
    mmcls==0.25.0 \
    mmdet==2.25.3 \
    mmdet3d==1.0.0rc6 \
    mmsegmentation==0.29.1
```

### 6.4 WorldEngine リポジトリ

```bash
git clone https://github.com/OpenDriveLab/WorldEngine.git ~/WorldEngine

# NAVSIM devkit v1.1 (評価ツールとして必須)
git clone -b v1.1 https://github.com/autonomousvision/navsim.git ~/navsim
```

### 6.5 AlgEngine 固有依存

```bash
cd ~/WorldEngine/projects/AlgEngine
pip install -r requirements.txt
pip install shapely==2.0.4
```

### 6.6 RL / 分散学習ツール

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

### 6.7 インストール確認

```bash
python - <<'EOF'
import torch, mmcv, mmdet3d
print(f"PyTorch : {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"mmcv    : {mmcv.__version__}")
print(f"mmdet3d : {mmdet3d.__version__}")
EOF
```

---

## 7. simengine 環境

WorldEngine SimEngine の 3DGS フォトリアルシミュレーションに使用する。  
gsplat は `--no-build-isolation` でビルドが必須。

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

### 7.2 gsplat v1.4.0 (Gaussian Splatting コアライブラリ)

既インストールの PyTorch を参照するため、`--no-build-isolation` が必須。

```bash
pip install ninja
pip install \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" \
    --no-build-isolation
```

### 7.3 Ray & Hydra

```bash
pip install \
    "ray[default]==2.9.3" \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    aiohttp aiohttp-cors prometheus_client
```

### 7.4 SimEngine リポジトリ & 依存

```bash
# WorldEngine がまだなければクローン
[ -d ~/WorldEngine ] || git clone https://github.com/OpenDriveLab/WorldEngine.git ~/WorldEngine

# MTGS (Multi-Traversal Gaussian Splatting)
git clone https://github.com/OpenDriveLab/MTGS.git ~/MTGS

[ -d ~/navsim ] || git clone -b v1.1 https://github.com/autonomousvision/navsim.git ~/navsim

cd ~/WorldEngine/projects/SimEngine
pip install -r requirements.txt

pip install \
    nuscenes-devkit==1.1.11 \
    numpy==1.24.4 \
    scipy pandas tqdm pillow \
    opencv-python-headless==4.8.1.78 \
    pyquaternion shapely==2.0.4
```

### 7.5 EGL ヘッドレスレンダリング設定

GPU サーバでディスプレイなしにレンダリングするため必要。

```bash
# /etc/environment または環境変数ファイルに追記
echo 'PYOPENGL_PLATFORM=egl' >> ~/.bashrc
echo 'EGL_DEVICE_ID=0'       >> ~/.bashrc
```

### 7.6 インストール確認

```bash
python - <<'EOF'
import torch, gsplat, ray
print(f"PyTorch : {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"gsplat  : {gsplat.__version__}")
print(f"ray     : {ray.__version__}")
EOF
```

---

## 8. 環境変数の永続化

`~/.bashrc` に追記し `source ~/.bashrc` で反映する。

```bash
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
```

各環境の `PYTHONPATH` は activate 時に自動設定されないため、
必要に応じてシェル関数として追記する。

```bash
# act-uniad2 を拡張した例
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
```

---

## 9. データセット準備

### 9.1 全体ディレクトリ構成

```
~/UniAD/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   └── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   └── nuscenes_infos_temporal_val.pkl
│   └── others/
│       └── motion_anchor_infos_mode6.pkl
└── work_dirs/

~/WorldEngine/
├── data/
│   ├── raw/
│   │   ├── nuscenes/          ← symlink or copy
│   │   └── nuplan/
│   │       ├── maps/
│   │       └── openscene/
│   ├── alg_engine/
│   │   ├── ckpts/
│   │   ├── merged_infos_navformer/
│   │   │   ├── nuplan_openscene_navtrain.pkl
│   │   │   └── nuplan_openscene_navtest.pkl
│   │   └── openscene-synthetic/
│   └── sim_engine/
│       └── scene_reconstructions/
```

### 9.2 UniAD 用 nuScenes info の取得

**方法A: HuggingFace から取得 (推奨)**

```bash
act-uniad2
cd ~/UniAD

mkdir -p data/infos data/others

wget -q -P data/infos \
    https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/nuscenes_infos_temporal_train.pkl \
    https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/nuscenes_infos_temporal_val.pkl

wget -q -P data/others \
    https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/motion_anchor_infos_mode6.pkl
```

**方法B: 自前生成**

```bash
cd ~/UniAD/data && mkdir infos
bash ../tools/uniad_create_data.sh
# → nuscenes_infos_temporal_{train,val}.pkl が生成される
# 注意: config の data_root を '' に変更すること (Issue #13)
```

### 9.3 WorldEngine 公式データセット (HuggingFace)

```bash
act-alge
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='OpenDriveLab/WorldEngine',
    repo_type='dataset',
    local_dir=f'{__import__("os").environ["WORLDENGINE_ROOT"]}/data/',
)
EOF
```

---

## 10. 稀少条件データセットの統合

| データセット | 主な特徴 | ライセンス |
|------------|---------|-----------|
| ACDC | 霧・雨・夜・雪 (実世界) | 研究用無償 |
| nuScenes-C | 腐敗バリアント 16 種 | Apache 2.0 |
| Adver-City | CARLA 合成 6 天候条件 | オープン |
| Bench2Drive | CARLA 44 シナリオ 2 M 枚 | Apache 2.0 |

### 10.1 ACDC 変換スクリプト

`~/UniAD/tools/data_converter/acdc_to_nuscenes.py`:

```python
"""ACDC → nuScenes 形式変換 (骨格実装)"""
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
            print(f"[WARN] {cond_dir} not found, skip.")
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
    p.add_argument('--out-dir', required=True)
    p.add_argument('--conditions', nargs='+',
                   default=['fog', 'night', 'rain', 'snow'])
    args = p.parse_args()
    main(args.acdc_root, args.out_dir, args.conditions)
```

```bash
act-uniad2
python tools/data_converter/acdc_to_nuscenes.py \
    --acdc-root /path/to/acdc \
    --out-dir data/acdc_nuscenes
```

### 10.2 nuScenes-C (腐敗バリアント生成)

```bash
act-uniad2
pip install nuscenes-c-tools

python -m nuscenes_c.generate \
    --data-root data/nuscenes \
    --out-dir data/nuscenes_c \
    --corruptions fog rain night motion_blur \
    --severity 1 2 3
```

---

## 11. データオーグメンテーション実装

`~/UniAD/projects/mmdet3d_plugin/uniad/utils/weather_augment.py`:

```python
"""全カメラに一貫した天候オーグメンテーションを適用する Transform"""
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RandomWeatherAugmentation:
    """
    マルチビューカメラへの一貫天候オーグメンテーション。
    全カメラに同じ天候条件を適用することで整合性を保つ。
    """

    def __init__(self, fog_prob=0.15, rain_prob=0.15,
                 night_prob=0.10, blur_prob=0.10, snow_prob=0.05):
        self.fog_prob   = fog_prob
        self.rain_prob  = rain_prob
        self.night_prob = night_prob
        self.blur_prob  = blur_prob
        self.snow_prob  = snow_prob

    def _add_fog(self, img, severity=0.5):
        """Koschmieder モデルに基づく霧シミュレーション"""
        h, w = img.shape[:2]
        depth = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
        t = np.exp(-severity * depth * 3.0)
        fog = np.array([210, 210, 210], dtype=np.float32)
        return np.clip(img.astype(np.float32) * t[..., None]
                       + fog * (1 - t[..., None]), 0, 255).astype(np.uint8)

    def _add_rain(self, img, intensity=800):
        rain = img.copy()
        h, w = img.shape[:2]
        for _ in range(intensity):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            length = np.random.randint(10, 30)
            angle  = np.random.uniform(-20, 20)
            x2 = int(x1 + length * np.sin(np.radians(angle)))
            y2 = int(y1 + length * np.cos(np.radians(angle)))
            cv2.line(rain, (x1, y1), (x2, y2), (200, 200, 200), 1)
        return cv2.addWeighted(img, 0.7, rain, 0.3, 0)

    def _add_night(self, img):
        gamma = np.random.uniform(2.5, 4.0)
        lut = np.array([(i / 255.0) ** gamma * 255
                        for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    def _add_motion_blur(self, img):
        k = np.random.choice([7, 11, 15])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = np.ones(k) / k
        return cv2.filter2D(img, -1, kernel)

    def _add_snow(self, img, intensity=500):
        snow = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        for _ in range(intensity):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.circle(snow, (x, y), np.random.randint(1, 3),
                       (255, 255, 255), -1)
        return np.clip(snow, 0, 255).astype(np.uint8)

    def __call__(self, results):
        r = np.random.random()
        aug_func = None
        cumprob = 0
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
            return results
        results['img'] = [aug_func(img) for img in results.get('img', [])]
        return results
```

---

## 12. 2 段階学習パイプライン (UniAD)

```bash
act-uniad2
cd ~/UniAD
```

### 12.1 Stage1: Perception 学習 (20 エポック)

```bash
./tools/dist_train.sh projects/configs/stage1_track_map/base_track_map.py 8  --work-dir work_dirs/stage1_base
```

学習するモジュール: BEVFormer / TrackHead / MapHead

### 12.2 Stage2: End-to-End 学習 (6 エポック)

```bash
# 標準設定
./tools/dist_train.sh projects/configs/stage2_e2e/base_e2e.py 8 --work-dir work_dirs/stage2_e2e

# 稀少条件オーグメンテーション付き
./tools/dist_train.sh projects/configs/stage2_e2e/base_e2e_adverse.py 8 --work-dir work_dirs/stage2_e2e_adverse
```

学習するモジュール: 全 Head (TrackHead / MapHead / MotionHead / OccHead / PlanningHead)

### 12.3 学習スケジュール設定

```python
# config 内
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
runner = dict(type='EpochBasedRunner', max_epochs=20)  # Stage1
# Stage2 は max_epochs=6
```

### 12.4 再開・ミニ動作確認

```bash
# チェックポイントから再開
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e.py 8 \
    --resume-from work_dirs/stage2_e2e/epoch_3.pth

# nuScenes-mini で高速動作確認
./tools/dist_train.sh \
    projects/configs/stage2_e2e/base_e2e.py 1 \
    --cfg-options \
        data.train.ann_file='data/infos/nuscenes_infos_temporal_train_mini.pkl' \
        data.val.ann_file='data/infos/nuscenes_infos_temporal_val_mini.pkl' \
        runner.max_epochs=2 \
        evaluation.interval=1
```

---

## 13. WorldEngine ポストトレーニングパイプライン

### 13.1 Phase 2-A: Open-loop 評価 & 希少ケース抽出

```bash
act-alge
cd $ALGENGINE_ROOT

# Open-loop 評価 (navtest 全体)
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_uniad_50pct.py \
    work_dirs/e2e_uniad_50pct/epoch_20.pth \
    8
# → work_dirs/e2e_uniad_50pct/navtest.csv

# 希少ケース抽出 (衝突・逸脱・低進行度)
python scripts/rare_case_sampling_by_pdms.py \
    --pdm-result work_dirs/e2e_uniad_50pct/navtest.csv \
    --base-split configs/navsim_splits/navtest_split/navtest.yaml \
    --output-dir configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare
```

抽出されるスプリット:
```
configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare/
├── navtest_collision.yaml    ← no_at_fault_collisions < 1.0
├── navtest_off_road.yaml     ← drivable_area_compliance < 1.0
└── navtest_ep_1pct.yaml      ← ego_progress 下位 1%
```

### 13.2 Phase 2-B: SimEngine ロールアウト生成

```bash
act-sime
cd $SIMENGINE_ROOT

# 動作確認 (事前学習済みモデルで 288 シナリオ評価)
bash scripts/closed_loop_test.sh

# 分散ロールアウト生成 (希少シナリオから)
bash scripts/run_ray_distributed_rollout.sh \
    $ALGENGINE_ROOT/configs/worldengine/e2e_uniad_50pct.py \
    $ALGENGINE_ROOT/work_dirs/e2e_uniad_50pct/epoch_20.pth \
    e2e_uniad_50pct \
    navtrain_50pct_collision \
    navtrain

# AlgEngine 形式に変換
python scripts/export_simulation_data.py \
    --test_path experiments/closed_loop_exps/e2e_uniad_50pct/navtrain_NR \
    --appendix $(date +%Y%m%d)
```

### 13.3 Phase 2-C: RL ファインチューニング

```bash
act-alge
cd $ALGENGINE_ROOT

./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    8 \
    work_dirs/e2e_uniad_50pct/epoch_20.pth
```

RL fine-tune config のポイント:

```python
# configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py
_base_ = ['./e2e_uniad_50pct.py']

data = dict(
    train=dict(
        scenario_filter=[
            'configs/navsim_splits/navtrain_split/e2e_uniad_50pct_ep20/'
            'navtrain_50pct_collision.yaml',
            'configs/navsim_splits/navtrain_split/e2e_uniad_50pct_ep20/'
            'navtrain_50pct_off_road.yaml',
        ],
        extra_ann_files=[
            'data/alg_engine/openscene-synthetic/meta_datas/',
        ],
    )
)

optimizer = dict(type='AdamW', lr=5e-5)   # 事前学習の 1/4
runner    = dict(type='EpochBasedRunner', max_epochs=8)
```

---

## 14. 評価・ベンチマーク

### 14.1 UniAD 標準評価 (nuScenes val)

```bash
act-uniad2
cd ~/UniAD

./tools/dist_test.sh \
    projects/configs/stage2_e2e/base_e2e.py \
    ckpts/uniad_base_e2e.pth \
    8 --eval bbox

# 期待性能 (UniAD-B, R101):
# Tracking AMOTA : 0.380
# Mapping IoU    : 0.314
# Motion minADE  : 0.794
# Occ IoU-n      : 64.0
# Planning Col.  : 0.29%
```

### 14.2 悪天候ロバスト性評価 (nuScenes-C)

```bash
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

### 14.3 WorldEngine 閉ループ評価

```bash
act-sime
cd $ALGENGINE_ROOT

bash scripts/run_ray_distributed_testing.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
    e2e_uniad_rlft \
    navtest_failures \
    NR

# 期待性能 (WorldEngine フルパイプライン後):
# CL 成功率 : 88.89%
# CL PDMS*  : 70.12
```

### 14.4 Bench2Drive 閉ループ評価

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

## 15. トラブルシューティング

### `AssertionError: data_root is not empty`

自前生成の pkl に絶対パスが埋め込まれている。config で修正する:

```python
data = dict(train=dict(data_root='', ...))
```

### `CUDA out of memory`

BEV 集約フレーム数を削減する:

```python
model = dict(
    img_bev_encoder_backbone=dict(queue_length=3)  # 5 → 3
)
```

または gradient checkpointing を有効にする:

```python
model = dict(img_backbone=dict(with_cp=True))
```

### mmcv バージョン不整合

```bash
# uniad2 環境: ホイール再インストール
pip install mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    --force-reinstall

# algengine 環境: ソースビルドからやり直し
cd ~/mmcv
pip install -v -e . --force-reinstall
```

### gsplat ビルド失敗

```bash
# CUDA Toolkit のヘッダを確認
ls /usr/local/cuda/include/cuda_runtime.h

# --no-build-isolation を必ず付ける
pip install "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" \
    --no-build-isolation --force-reinstall
```

### Stage1 の AMOTA が論文値より低い

```python
# base_track_map.py で確認・修正
loss_past_traj = False   # False に設定
# img_neck と BN パラメータが unfreeze になっていること
```

### Ray 分散シミュレーションがハング

```bash
ray stop --force
pkill -f "ray::"
# 再起動
ray start --head --num-gpus=8
```

---

## 付録: 推奨ディレクトリ構成 (統合版)

```
~/
├── venvs/
│   ├── uniad2/       ← source ~/venvs/uniad2/bin/activate
│   ├── algengine/    ← source ~/venvs/algengine/bin/activate
│   └── simengine/    ← source ~/venvs/simengine/bin/activate
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
├── mmcv/             ← algengine 用ソースビルド
├── navsim/           ← v1.1
└── MTGS/             ← simengine 用 3DGS
```

---

## 参考リンク

- [UniAD GitHub (v2.0)](https://github.com/OpenDriveLab/UniAD/tree/v2.0)
- [UniAD HuggingFace](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes)
- [WorldEngine GitHub](https://github.com/OpenDriveLab/WorldEngine)
- [WorldEngine HuggingFace Dataset](https://huggingface.co/datasets/OpenDriveLab/WorldEngine)
- [AlgEngine 使用ガイド](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/algengine_usage.md)
- [SimEngine 使用ガイド](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/simengine_usage.md)
- [MTGS](https://github.com/OpenDriveLab/MTGS)
- [NAVSIM v1.1](https://github.com/autonomousvision/navsim)
- [nuScenes 公式](https://www.nuscenes.org/download)
- [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [nuScenes-C](https://github.com/Daniel-xsy/RoboBEV)
