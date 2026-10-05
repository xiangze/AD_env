# UniAD を Docker で動かす — セットアップから推論まで

単一 GPU(例: RTX 3080 Ti 12GB)のワークステーションで、Docker / docker compose を使って
UniAD v2.0 の推論(open-loop 評価)を実行するまでの手順。CUDA 13 世代の新しいドライバを
積んだホストでも、後方互換で CUDA 11.8 ベースのコンテナを動かせる。

> この手順は nuScenes **v1.0-mini** での動作確認(配管確認)を主眼にする。
> フル trainval 評価には数百 GB のディスクが必要。まず mini で通し、容量が確保でき次第
> trainval に切り替える。

---

## 1. ホスト前提条件

| 項目 | 要件 |
| --- | --- |
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA(compute capability 8.6 = RTX 30 系で確認済み) |
| ドライバ | CUDA 11.8 以上に対応するもの(例: 525+。580 系 / CUDA 13 表示でも可) |
| Docker | **apt 版 Docker CE**(snap 版は不可。後述) |
| ディスク | mini 動作確認なら数十 GB、trainval なら 1TB NVMe 推奨 |

GPU の compute capability は次で確認する:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

`8.6` 以下ならこの手順のイメージ(CUDA 11.8 / torch 2.0.1)でネイティブに動く。
`nvidia-smi` が「CUDA Version: 13.0」と表示しても、それは**ホストドライバが対応する上限**で
あり、CUDA は後方互換なので 11.8 コンテナは問題なく動く。

### 1.1 Docker は apt 版を使う(重要)

**snap 版 Docker は使わない。** snap の confinement が NVIDIA の GPU 注入
(`nvidia-cuda-mps-control` のマウント、`libnvidia-ml.so.1` の解決)を壊し、
`--gpus all` が機能しない。apt 版 Docker CE へ移行する:

```bash
# snap 版が入っている場合は削除(イメージは事前に docker save で退避)
sudo snap remove docker

# 公式 apt から Docker CE
sudo install -m0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER   # 反映は再ログイン
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

GPU 注入が効くか、素の CUDA イメージで切り分ける:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

コンテナ内 `nvidia-smi` にホストと同じ GPU が出れば成功。

---

## 2. ディレクトリと環境変数

ホスト側にデータレイクと ckpt 置き場を用意する(nuScenes は UniAD/VAD などで共用)。

```
/mnt/data/
├── downloads/        # 手動DLのアーカイブ置き場
├── nuscenes/         # 展開済み生データ(ro 共用)
├── infos/uniad/      # UniAD 用 info pkl(生成先 rw)
├── others/           # motion anchor(rw)
└── work_dirs/uniad/  # 評価ログ・結果(rw)
/mnt/ckpts/uniad/     # 学習済み重み(ro)
```

`.env`(docker-compose.yml と同じディレクトリ):

```dotenv
HOST_DATA_ROOT=/mnt/data
HOST_CKPT_DIR=/mnt/ckpts
UNIAD_GPUS=0
```

---

## 3. データと重みの準備(手動DL + 自動展開)

nuScenes はログインが必要なため、アーカイブの取得だけは手動。以下3点を
`/mnt/data/downloads/` に置く(nuScenes 公式サイトから取得):

- `v1.0-mini.tgz` … mini 本体(samples/sweeps/maps を含む)
- `can_bus.zip` … CAN bus expansion(UniAD は自車運動に必須)
- `nuScenes-map-expansion-v1.3.zip` … ベクターマップ(MapHead が使用)

学習済み重みは public なので自動取得できる:

```bash
mkdir -p /mnt/ckpts/uniad
wget -O /mnt/ckpts/uniad/uniad_base_e2e.pth \
  https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
```

展開と info 生成は `prepare_mini_data.sh`(冪等)で自動化する:

```bash
DATA_ROOT=/mnt/data ./prepare_mini_data.sh
```

このスクリプトは、生データ展開(mini / can_bus / map expansion)→ motion anchor 取得 →
コンテナ内で `create_data.py --version v1.0-mini` による info pkl 生成、までを行う。
展開後のレイアウト:

```
/mnt/data/nuscenes/
├── v1.0-mini/          # メタデータ JSON(scene.json 等)
├── samples/ sweeps/
├── maps/
│   └── expansion/      # boston-seaport.json 等(MapHead 必須)
└── can_bus/
```

---

## 4. イメージのビルド

UniAD は共通ベースイメージ `uniad-worldengine-base` の上に積むため、**先にベースをビルド**する。

```bash
make build-base        # または: docker build -f Dockerfile.base -t uniad-worldengine-base:latest .
make build-uniad2
```

### ビルド上の要点(Dockerfile.uniad2 に焼き込み済み)

- **mmcv の CUDA op**: Python 3.9 では `mmcv-full` のビルド済み wheel が取得できず、
  ソースビルドになる。`--no-build-isolation` を付けて `pkg_resources` 不在を回避し、
  `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6` で op をイメージに焼き込む。これを怠ると
  実行時に `ModuleNotFoundError: No module named 'mmcv._ext'` になる。
- **NumPy は 1.x 固定**: 焼き込み後に `numpy==1.23.4` を固定する。2.x が混入すると
  `numpy.core.multiarray failed to import` でこける。
- 焼き込んであるので、コンテナを作り直しても `_ext` は失われない。

ベースイメージが出来ているか確認:

```bash
docker images | grep uniad-worldengine-base
```

---

## 5. コンテナ起動

```bash
make up-uniad2
docker ps              # uniad2 が Up であること
```

> compose の uniad2 は `command: sleep infinity` で常駐させる。`/bin/bash` のままだと
> `up -d` で TTY が無く即終了し、後続の `exec` が "not running" になる。

ckpt とデータがコンテナから見えるか確認:

```bash
docker compose --env-file .env exec uniad2 bash -c \
  "cd /workspace/UniAD && ls ckpts/ data/nuscenes/ data/infos/ data/others/"
```

---

## 6. 推論(open-loop 評価)の実行

UniAD の評価スクリプトは `uniad_dist_eval.sh`(`dist_test.sh` ではない)。
単一 GPU なので引数は `1`。ログは work_dirs に残す。

```bash
make eval-uniad-openloop
# 実体:
docker compose --env-file .env exec uniad2 bash -c \
  "cd /workspace/UniAD && \
   ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py \
     ./ckpts/uniad_base_e2e.pth 1 \
     2>&1 | tee work_dirs/eval_uniad_$(date +%Y%m%d_%H%M%S).log"
```

tracking AMOTA、mapping IoU、motion minADE、occupancy IoU、planning L2/衝突率が出力される。

> mini(val 2 シーン)の数値は論文値とは無関係の**動作確認用**。正式な指標は trainval で。

### VRAM が足りない場合

12GB で OOM が出たら、config で BEV フレーム数を減らすか勾配チェックポイントを有効化する:

```python
# base_e2e.py
queue_length = 3                       # 既定 5 → 3
model = dict(img_backbone=dict(with_cp=True))
```

---

## 7. つまずきやすいエラーと対処

| 症状 | 原因 / 対処 |
| --- | --- |
| `NVIDIA Driver was not detected` | `--gpus all` 無し、または Toolkit 未設定。§1.2 を実施 |
| `nvidia-cuda-mps-control ... no such file` / `libnvidia-ml.so.1 cannot open` | snap 版 Docker。apt 版へ移行(§1.1) |
| `pull access denied: uniad-worldengine-base` | ベース未ビルド。先に `make build-base` |
| `No module named 'mmcv._ext'` | op 未焼き込み。Dockerfile の mmcv 行を `--no-build-isolation` で再ビルド |
| `numpy.core.multiarray failed to import` | NumPy 2.x 混入。`numpy==1.23.4` に固定 |
| `service "uniad2" is not running` | `command` が `/bin/bash` で即終了。`sleep infinity` に |
| `ckpts/uniad_base_e2e.pth not found` | ホストの `/mnt/ckpts/uniad/` に重みを置く |
| `Database version not found: .../v1.0-trainval` | config/pkl の version 不一致。mini なら info pkl を mini で再生成 |
| `maps/expansion/boston-seaport.json not found` | map expansion 未展開。§3 の zip を展開 |
| `No module named 'nuplan'` | AlgEngine 系の open-loop を実行している。UniAD 評価は `eval-uniad-openloop` を使う |

---

## 8. ログとデータの永続化

評価の標準出力は `tee work_dirs/...log` で、結果ファイルは config の `work_dir` 配下に出す。
`work_dirs` をホストへ bind mount しておけば、コンテナ再作成でも残り、他環境へコピーできる。

```yaml
# docker-compose.yml の uniad2 volumes
- type: bind
  source: /mnt/data/work_dirs/uniad
  target: /workspace/UniAD/work_dirs
  bind: { create_host_path: true }
```

---

## 付録: 最短手順まとめ

```bash
# 0. 前提: apt Docker CE + NVIDIA Container Toolkit(§1)
# 1. 手動DL 3点を /mnt/data/downloads へ、重みを /mnt/ckpts/uniad へ
# 2. データ展開 + info 生成
DATA_ROOT=/mnt/data ./prepare_mini_data.sh
# 3. ビルド
make build-base && make build-uniad2
# 4. 起動
make up-uniad2
# 5. 推論
make eval-uniad-openloop
```
