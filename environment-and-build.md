# AD_env — 環境・ビルド設計ガイド

UniAD / VAD / SparseDrive(v1, V2)/ Hydra-NeXt を、1台のワークステーション
(単一 GPU 想定, 検証環境: RTX 3080 Ti 12GB / compute capability 8.6)で Docker 運用する
ための設計と、ビルドで踏みやすい落とし穴の対処をまとめる。

各モデルの「導入手順」は `README_docker_jp.md` / `README_docker_en.md`(UniAD 中心)を参照。
本書は**モデルを跨ぐ設計判断と共通のハマりどころ**を扱う。

---

## 1. モデル × データトラック対応表

モデルは依存スタックと評価データで3つのトラックに分かれる。**生データを共有できるのは
同一トラック内だけ**で、info/anchor のフォーマットは全モデル非互換(各自生成)。

| モデル | トラック | 評価 | データ | Python/CUDA/torch | mmcv | 専用 Dockerfile |
| --- | --- | --- | --- | --- | --- | --- |
| UniAD v2.0 | nuScenes | open-loop | nuScenes | 3.9 / 11.8 / 2.0.1 | 1.6.2(src) | Dockerfile.uniad2(+ base) |
| VAD | nuScenes | open-loop | nuScenes | 3.8 / 11.1 / 1.9.1 | 1.4.0(wheel) | Dockerfile.vad |
| SparseDrive v1 | nuScenes | open-loop | nuScenes | 3.8 / 11.6 / 1.13.0 | 1.7.1(wheel) | Dockerfile.sparsedrive |
| SparseDriveV2 | NAVSIM | PDMS/EPDMS | OpenScene + nuplan maps | 3.9 / 11.8 / 2.0.1 | なし(nuplan-devkit) | Dockerfile.sparsedrivev2 |
| Hydra-NeXt | CARLA | closed-loop | Bench2Drive(CARLA 0.9.15) | 3.8 / 11.8 / 2.0.1 | bundled | Dockerfile.closedloop |

- **nuScenes トラック**(UniAD/VAD/SparseDrive v1): `/mnt/data/nuscenes` を ro 共用。
- **NAVSIM トラック**(SparseDriveV2): OpenScene(navsim_logs/sensor_blobs)+ nuplan maps。
  nuScenes とは別データ。WorldEngine AlgEngine と同系統。
- **CARLA トラック**(Hydra-NeXt): CARLA サーバ + Bench2Drive。nuScenes も NAVSIM も使わない。

---

## 2. 共有データレイク設計

生データは1コピー、info/anchor と出力だけモデル別に分ける。

```
/mnt/data/
├── downloads/              # 手動DLアーカイブ(nuScenes はログイン必須)
├── nuscenes/               # 共有生データ(UniAD/VAD/SparseDrive v1 が ro 共用)
│   ├── v1.0-mini/ (or v1.0-trainval/)
│   ├── samples/ sweeps/ maps/expansion/ can_bus/
├── openscene/              # NAVSIM トラック(SparseDriveV2)
│   ├── navsim_logs/ sensor_blobs/ maps/
├── infos/{uniad,vad,sparsedrive}/   # モデル別 info pkl(rw)
├── others/                 # UniAD motion anchor
├── navsim_exp/             # NAVSIM キャッシュ・実験出力(rw)
└── work_dirs/{uniad,vad,...}/       # 評価ログ・結果(rw)
/mnt/ckpts/{uniad,vad,sparsedrive,sparsedrivev2,hydranext}/   # 重み(ro)
```

compose は `-f docker-compose.yml -f compose.<model>.yml` の重ね合わせで、各モデルの
マウントを足す。生データは `read_only: true`、生成物・ログは rw(`create_host_path: true`)。

---

## 3. ホスト基盤(全モデル共通)

### 3.1 Docker は apt 版(snap 不可)

snap 版 Docker は confinement が NVIDIA GPU 注入(`nvidia-cuda-mps-control` マウント、
`libnvidia-ml.so.1` 解決)を壊し `--gpus all` が機能しない。**apt 版 Docker CE へ移行**する
(手順は README 参照)。これは全モデルに効く前提条件。

### 3.2 NVIDIA Container Toolkit + 後方互換

`nvidia-ctk runtime configure --runtime=docker` 後、素の CUDA イメージで注入を検証。
ホストドライバが CUDA 13 対応でも、後方互換で 11.x / 11.8 コンテナは動く。
`nvidia-smi` の「CUDA Version」はドライバ上限であり、コンテナ CUDA ではない。

### 3.3 GPU アーキ制約

compute capability 8.6(RTX 30 系)までは本書の各イメージでネイティブ動作。
8.9(Ada)以降は、cu111/cu116 系(VAD / SparseDrive v1)のカーネルを持たないため、
より新しい CUDA ベースへの移植が必要。全 Dockerfile に `TORCH_CUDA_ARCH_LIST="8.6"` を設定。

---

## 4. ビルドの共通ハマりどころ

このプロジェクトで実際に踏んだ順に、原因と恒久対処を記す。**いずれも Dockerfile に
焼き込むのが鉄則**(実行中コンテナでの手当ては再作成で消える)。

### 4.1 `mmcv._ext` が無い

mmcv の CUDA op が未コンパイル。対処はトラックで異なる:

- **wheel が使える場合**(VAD=1.4.0/cu111, SparseDrive v1=1.7.1/cu116, いずれも cp38):
  openmmlab のインデックス指定で wheel を入れれば `_ext` 同梱。Python を 3.8 に保つのが条件。
- **wheel が無い場合**(UniAD: mmcv 1.6.2 × cu118 × **cp39** の wheel が無い):
  ソースビルドになる。`--no-build-isolation` を付け、`FORCE_CUDA=1
  TORCH_CUDA_ARCH_LIST=8.6` で op を焼き込む。検証 import は `cd /tmp` してから
  (ソースツリー `/build/mmcv` 内だと生ソースを読んで `_ext` 不在になる)。

### 4.2 `pkg_resources` 不在でビルド失敗

`--no-build-isolation` 無しのソースビルドで、隔離環境に setuptools が無く
`ModuleNotFoundError: No module named 'pkg_resources'`。→ `--no-build-isolation` を付け、
イメージ側に `setuptools` / `wheel` / `ninja` を用意しておく。

### 4.3 NumPy 2.x による ABI 不整合

torch/mmcv の拡張は NumPy 1.x でビルドされており、実行時 NumPy が 2.x だと
`numpy.core.multiarray failed to import`。`--force-reinstall` が numpy を引き上げるのが
主因。→ 各 Dockerfile 末尾で `numpy==1.23.x` を `--no-deps` 固定。

### 4.4 自前 CUDA op のビルド

mmcv とは別に、各モデルが独自 op を持つ:

- UniAD: mmdet3d 0.17/1.0 系の op(ソースビルド)
- SparseDrive v1: Deformable Aggregation(`projects/mmdet3d_plugin/ops` で setup.py develop)
- いずれも `FORCE_CUDA=1` + `TORCH_CUDA_ARCH_LIST=8.6` で焼き込み。

### 4.5 flash-attn(SparseDrive 系)

ビルドが時間・メモリ食いでこけやすい。`MAX_JOBS=4`(RAM 逼迫時は 2)。版は repo の
`requirement.txt` に合わせる。

### 4.6 ベースイメージ未ビルド

UniAD/algengine は `FROM uniad-worldengine-base`。未ビルドだと Docker Hub を引きに行き
`pull access denied`。→ `make build-base` を先に。

### 4.7 コンテナ即終了

compose の `command: /bin/bash` は `up -d` で TTY 不在のため即 Exit し、後続 `exec` が
`service not running`。→ `command: sleep infinity`(exec で使うサービスは常駐させる)。

---

## 5. データ準備の共通点

### 5.1 nuScenes(手動DL 3点)

ログイン必須のため自動化不可。`/mnt/data/downloads/` に:
`v1.0-mini.tgz`(or trainval)/ `can_bus.zip` / `nuScenes-map-expansion-v1.3.zip`。
展開と info 生成は `prepare_data.sh <model>`(冪等)。

- info の version は **pkl の `metadata['version']` 由来**。config ではなく pkl が trainval/mini を決める。
- UniAD は map expansion(`maps/expansion/*.json`)と can_bus が必須。
- mini は converter に `--version v1.0-mini` を渡す(UniAD は `tools/create_data.py` が正規ルート)。

### 5.2 容量

trainval は数百 GB。内蔵ディスクが逼迫する場合は外部ストレージを `/mnt/data` にマウント、
または mini で配管確認 → 容量確保後に trainval。mini の指標は論文値と無関係(動作確認用)。

---

## 6. 評価コマンドの型(トラック別)

```bash
# nuScenes open-loop
#   UniAD: 専用ラッパ(dist_test.sh ではない), 1 GPU
./tools/uniad_dist_eval.sh projects/configs/stage2_e2e/base_e2e.py ckpts/uniad_base_e2e.pth 1
#   VAD / SparseDrive v1: mmdet3d test, 必ず 1 GPU 非分散
python tools/test.py <config> <ckpt> --launcher none --eval bbox

# NAVSIM(SparseDriveV2): キャッシュ → PDMS 評価
sh scripts/cache/run_dataset_caching_navtest.sh
sh scripts/evaluation/run_pdm_score_navtest_v2.sh

# CARLA(Hydra-NeXt): CARLA サーバ起動後、Bench2Drive leaderboard
TEAM_AGENT=hydra_next_agent.py TEAM_CONFIG=<eval_cfg>+<ckpt> ALGO=hydranext \
  bash leaderboard/scripts/run_evaluation_multi_vad.sh
```

分散評価(複数 GPU)は VAD/SparseDrive で結果が不正確になるため、評価は 1 GPU 非分散。

---

## 7. ファイル一覧

| ファイル | 役割 |
| --- | --- |
| Dockerfile.base | UniAD/algengine 共通ベース(CUDA 11.8 / torch 2.0.1) |
| Dockerfile.uniad2 | UniAD v2.0 |
| Dockerfile.vad | VAD(独立スタック cu111) |
| Dockerfile.sparsedrive | SparseDrive v1(cu116, nuScenes) |
| Dockerfile.sparsedrivev2 | SparseDriveV2(NAVSIM) |
| Dockerfile.closedloop | Hydra-NeXt(CARLA 0.9.15 + Bench2Drive) |
| compose.multimodel.yml | 共有データレイク + 3モデル + CARLA の重ね合わせ |
| compose.sparsedrive(v2).yml | SparseDrive 各版のサービス |
| prepare_data.sh | 生データ展開 + モデル別 info 生成(raw\|uniad\|vad\|sparsedrive\|all) |
| Makefile.* | 各モデルのビルド/起動/評価ターゲット |

---

## 8. バージョン固定の方針

再現性のため、各 Dockerfile の `ARG *_REF` は追跡ブランチ(main/master)ではなく、
**検証済みコミット SHA** に固定することを推奨。torch / mmcv / numpy / flash-attn は
本書の表の値で固定済み。CARLA は 0.9.15 固定。
