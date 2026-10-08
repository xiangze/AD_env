"""SparseDriveV2 / NAVSIM 環境の動作確認。

    python /workspace/tools/verify_env.py

GPU があれば deformable_aggregation の CUDA カーネルを実際に 1 回呼び出す。
ckpt があれば SparseDriveV2 エージェントを構築して重みをロードする。
"""
import os
import sys
from pathlib import Path

ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f"[ OK ] {name}" + (f": {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def _torch():
    import torch

    s = f"torch {torch.__version__} (cuda {torch.version.cuda}), cuda available={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        s += f", device={torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}"
    return s


def _libs():
    import numpy
    import pytorch_lightning
    import nuplan  # noqa: F401
    import timm
    import hydra  # noqa: F401

    return f"numpy {numpy.__version__}, lightning {pytorch_lightning.__version__}, timm {timm.__version__}"


def _ops():
    import torch
    from navsim.agents.sparsedrive.ops import deformable_aggregation as da

    s = f"ext loaded ({Path(da.deformable_aggregation_ext.__file__).name})"
    if torch.cuda.is_available():
        # 小さい入力でカーネルを 1 回実行
        # shapes (kernel 定義より): feat [bs,cams,feat,C], shape [scale,2], start [scale],
        #                         loc [bs,pts,cams,2], weights [bs,pts,cams,scale,groups]
        bs, n_cams, c, n_pts, n_groups = 2, 3, 32, 16, 8
        shapes = [(8, 8), (4, 4)]
        n_feat = sum(h * w for h, w in shapes)
        feat = torch.randn(bs, n_cams, n_feat, c, device="cuda")
        shape = torch.tensor(shapes, device="cuda", dtype=torch.int32)
        start = torch.tensor([0, shapes[0][0] * shapes[0][1]], device="cuda", dtype=torch.int32)
        loc = torch.rand(bs, n_pts, n_cams, 2, device="cuda")
        weights = torch.rand(bs, n_pts, n_cams, len(shapes), n_groups, device="cuda")
        out = da.DeformableAggregationFunction.apply(feat, shape, start, loc, weights)
        s += f", CUDA kernel run -> {tuple(out.shape)}"
    return s


def _env():
    keys = ["NAVSIM_DEVKIT_ROOT", "NAVSIM_EXP_ROOT", "OPENSCENE_DATA_ROOT", "NUPLAN_MAPS_ROOT", "NUPLAN_MAP_VERSION"]
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"unset: {missing}")
    return ", ".join(f"{k}={os.environ[k]}" for k in keys[:3])


def _files():
    root = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "."))
    data = Path(os.environ.get("OPENSCENE_DATA_ROOT", "."))
    want = {
        "ckpt/resnet34.bin": root / "ckpt/resnet34.bin",
        "anchors": root / "ckpt/kmeans/trajectory_1024_256.npz",
        "ckpt v1": root / "ckpt/sparsedrive_navsimv1.ckpt",
        "ckpt v2": root / "ckpt/sparsedrive_navsimv2.ckpt",
        "maps": data / "maps",
        "navsim_logs/test": data / "navsim_logs/test",
        "navsim_logs/trainval": data / "navsim_logs/trainval",
    }
    return ", ".join(f"{k}:{'yes' if p.exists() else 'no'}" for k, p in want.items())


def _agent():
    root = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "."))
    ckpt = root / "ckpt/sparsedrive_navsimv1.ckpt"
    if not (ckpt.exists() and (root / "ckpt/resnet34.bin").exists()):
        return "skipped (run tools/download_weights.sh first)"
    os.chdir(root)
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = str(root / "navsim/planning/script/config/common/agent")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="sparsedrive_agent", overrides=[f"checkpoint_path={ckpt}"])
    agent = instantiate(cfg)
    agent.initialize()
    n = sum(p.numel() for p in agent.parameters()) / 1e6
    if torch.cuda.is_available():
        agent.cuda()
    return f"{type(agent).__name__} loaded, {n:.1f}M params"


check("torch / CUDA", _torch)
check("python libs", _libs)
check("sparsedrive CUDA ops", _ops)
check("NAVSIM env vars", _env)
check("files", _files)
check("agent + checkpoint", _agent)
sys.exit(0 if ok else 1)
