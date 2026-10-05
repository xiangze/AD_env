# Running UniAD with Docker — From Setup to Inference

How to run UniAD v2.0 inference (open-loop evaluation) on a single-GPU workstation
(e.g. RTX 3080 Ti, 12 GB) using Docker / docker compose. A host with a newer
CUDA-13-generation driver can still run the CUDA 11.8 container thanks to backward
compatibility.

> This guide targets a smoke test on nuScenes **v1.0-mini** (verifying the plumbing).
> Full trainval evaluation needs hundreds of GB of disk. Get mini working first, then
> switch to trainval once storage is available.

---

## 1. Host prerequisites

| Item | Requirement |
| --- | --- |
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA (verified on compute capability 8.6 = RTX 30 series) |
| Driver | Supports CUDA 11.8+ (e.g. 525+; a 580-series / "CUDA 13" readout is fine) |
| Docker | **apt Docker CE** (the snap build does NOT work — see below) |
| Disk | Tens of GB for the mini smoke test; 1 TB NVMe recommended for trainval |

Check the GPU compute capability:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

`8.6` or lower runs natively on this guide's image (CUDA 11.8 / torch 2.0.1). Even if
`nvidia-smi` reports "CUDA Version: 13.0", that is the **maximum the host driver
supports**; CUDA is backward compatible, so the 11.8 container runs fine.

### 1.1 Use the apt Docker, not snap (important)

**Do not use the snap Docker.** Its confinement breaks NVIDIA GPU injection
(mounting `nvidia-cuda-mps-control`, resolving `libnvidia-ml.so.1`), so `--gpus all`
fails. Move to apt Docker CE:

```bash
# If snap Docker is installed, remove it (export images first with docker save)
sudo snap remove docker

# Docker CE from the official apt repo
sudo install -m0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER   # re-login to apply
```

### 1.2 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU injection with a plain CUDA image:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

Seeing the same GPU inside the container means success.

---

## 2. Directories and environment variables

Prepare a shared data lake and a checkpoint directory on the host (nuScenes is shared
across UniAD/VAD/etc.).

```
/mnt/data/
├── downloads/        # manually downloaded archives
├── nuscenes/         # extracted raw data (shared, ro)
├── infos/uniad/      # UniAD info pkl (generated here, rw)
├── others/           # motion anchor (rw)
└── work_dirs/uniad/  # eval logs & results (rw)
/mnt/ckpts/uniad/     # trained weights (ro)
```

`.env` (next to docker-compose.yml):

```dotenv
HOST_DATA_ROOT=/mnt/data
HOST_CKPT_DIR=/mnt/ckpts
UNIAD_GPUS=0
```

---

## 3. Data and weights (manual download + automated extraction)

nuScenes requires login, so only the archive download is manual. Place these three
files under `/mnt/data/downloads/` (from the official nuScenes site):

- `v1.0-mini.tgz` — mini body (includes samples/sweeps/maps)
- `can_bus.zip` — CAN bus expansion (UniAD needs ego motion)
- `nuScenes-map-expansion-v1.3.zip` — vector maps (used by the MapHead)

The trained weights are public and can be fetched automatically:

```bash
mkdir -p /mnt/ckpts/uniad
wget -O /mnt/ckpts/uniad/uniad_base_e2e.pth \
  https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
```

Extraction and info generation are automated (idempotent) by `prepare_mini_data.sh`:

```bash
DATA_ROOT=/mnt/data ./prepare_mini_data.sh
```

It extracts the raw data (mini / can_bus / map expansion), downloads the motion anchor,
and generates the info pkl via `create_data.py --version v1.0-mini` inside the container.
Resulting layout:

```
/mnt/data/nuscenes/
├── v1.0-mini/          # metadata JSON (scene.json, etc.)
├── samples/ sweeps/
├── maps/
│   └── expansion/      # boston-seaport.json, etc. (required by MapHead)
└── can_bus/
```

---

## 4. Building the images

UniAD is layered on the shared base image `uniad-worldengine-base`, so **build the base
first**.

```bash
make build-base        # or: docker build -f Dockerfile.base -t uniad-worldengine-base:latest .
make build-uniad2
```

### Build notes (already baked into Dockerfile.uniad2)

- **mmcv CUDA ops**: on Python 3.9 no prebuilt `mmcv-full` wheel is available, so it
  builds from source. Pass `--no-build-isolation` to avoid the `pkg_resources` failure,
  and set `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6` to bake the ops into the image.
  Skipping this causes `ModuleNotFoundError: No module named 'mmcv._ext'` at runtime.
- **Pin NumPy to 1.x**: pin `numpy==1.23.4` after the mmcv install. NumPy 2.x causes
  `numpy.core.multiarray failed to import`.
- Because the ops are baked in, `_ext` survives container recreation.

Confirm the base image exists:

```bash
docker images | grep uniad-worldengine-base
```

---

## 5. Starting the container

```bash
make up-uniad2
docker ps              # uniad2 should be Up
```

> The uniad2 service uses `command: sleep infinity` to stay resident. With `/bin/bash`
> it would exit immediately under `up -d` (no TTY), and later `exec` would report
> "not running".

Check that ckpt and data are visible inside the container:

```bash
docker compose --env-file .env exec uniad2 bash -c \
  "cd /workspace/UniAD && ls ckpts/ data/nuscenes/ data/infos/ data/others/"
```

---

## 6. Running inference (open-loop evaluation)

UniAD's eval script is `uniad_dist_eval.sh` (NOT `dist_test.sh`). On a single GPU the
argument is `1`. Keep the log under work_dirs.

```bash
make eval-uniad-openloop
# equivalent:
docker compose --env-file .env exec uniad2 bash -c \
  "cd /workspace/UniAD && \
   ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py \
     ./ckpts/uniad_base_e2e.pth 1 \
     2>&1 | tee work_dirs/eval_uniad_$(date +%Y%m%d_%H%M%S).log"
```

It reports tracking AMOTA, mapping IoU, motion minADE, occupancy IoU, and planning
L2 / collision rate.

> mini (2 val scenes) numbers are a **smoke test only** and unrelated to paper values.
> Use trainval for real metrics.

### If VRAM is insufficient

On 12 GB, if you hit OOM, reduce the number of aggregated BEV frames or enable gradient
checkpointing in the config:

```python
# base_e2e.py
queue_length = 3                       # default 5 -> 3
model = dict(img_backbone=dict(with_cp=True))
```

---

## 7. Common errors and fixes

| Symptom | Cause / fix |
| --- | --- |
| `NVIDIA Driver was not detected` | Missing `--gpus all` or Toolkit not configured. Do §1.2 |
| `nvidia-cuda-mps-control ... no such file` / `libnvidia-ml.so.1 cannot open` | snap Docker. Move to apt (§1.1) |
| `pull access denied: uniad-worldengine-base` | Base not built. Run `make build-base` first |
| `No module named 'mmcv._ext'` | Ops not baked. Rebuild the mmcv step with `--no-build-isolation` |
| `numpy.core.multiarray failed to import` | NumPy 2.x present. Pin `numpy==1.23.4` |
| `service "uniad2" is not running` | `command` is `/bin/bash` and exits. Use `sleep infinity` |
| `ckpts/uniad_base_e2e.pth not found` | Put the weight in host `/mnt/ckpts/uniad/` |
| `Database version not found: .../v1.0-trainval` | config/pkl version mismatch. Regenerate info pkl for mini |
| `maps/expansion/boston-seaport.json not found` | Map expansion not extracted. Extract the zip in §3 |
| `No module named 'nuplan'` | You ran an AlgEngine-style open-loop. Use `eval-uniad-openloop` for UniAD |

---

## 8. Persisting logs and data

Send eval stdout through `tee work_dirs/...log`, and result files go under the config's
`work_dir`. Bind-mounting `work_dirs` to the host keeps them across container recreation
and lets you copy them elsewhere.

```yaml
# uniad2 volumes in docker-compose.yml
- type: bind
  source: /mnt/data/work_dirs/uniad
  target: /workspace/UniAD/work_dirs
  bind: { create_host_path: true }
```

---

## Appendix: shortest path

```bash
# 0. Prereqs: apt Docker CE + NVIDIA Container Toolkit (§1)
# 1. Put the 3 manual downloads in /mnt/data/downloads, weights in /mnt/ckpts/uniad
# 2. Extract data + generate info
DATA_ROOT=/mnt/data ./prepare_mini_data.sh
# 3. Build
make build-base && make build-uniad2
# 4. Start
make up-uniad2
# 5. Inference
make eval-uniad-openloop
```
