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

