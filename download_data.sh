# pretrained uniad 
mkdir -p ./ckpts/uniad
BASE=https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main
wget -O ./ckpts/uniad/uniad_base_e2e.pth $BASE/ckpts/uniad_base_e2e.pth

# data
mkdir -p ./data/uniad_infos/infos ./data/uniad_infos/others
wget -O ./data/uniad_infos/infos/nuscenes_infos_temporal_val.pkl \
  $BASE/data/nuscenes_infos_temporal_val.pkl
wget -O ./data/uniad_infos/others/motion_anchor_infos_mode6.pkl \
  $BASE/data/motion_anchor_infos_mode6.pkl

## VAD
# 1) can_bus 拡張(nuScenes 公式から取得し ./data/can_bus に展開)
#    nuScenes のダウンロードページの "CAN bus expansion" を取得
# 2) VAD-Base 重み。hustvl/VAD の README モデル動物園のリンク(Google Drive)を使う。
#    Google Drive は wget 不可なので gdown を使用:
pip install -U gdown
mkdir -p ./ckpts/vad
# README のリンクから FILE_ID を取得して:
gdown "https://drive.google.com/uc?id=<VAD_base の FILE_ID>" -O ./ckpts/vad/VAD_base.pth
# resnet50 バックボーンも同じ場所に(マウントで隠れる分の補填)
wget -O ./ckpts/vad/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth
# 3) vad_*.pkl は make prepare-vad-data で生成