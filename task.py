"""
task.py — MyAD / UniAD Vertex AI エントリポイント

Vertex AI Custom Job から呼び出される。
環境変数でタスク・モードを切り替えて評価スクリプトを実行し、
結果を GCS にアップロードする。

環境変数:
    AIP_MODEL_DIR   : 結果の GCS 保存先 (Vertex AI が自動設定)
    EVAL_TASKS      : all | tracking | mapping | motion | occupancy | planning
    EVAL_MODE       : normal | corruption
    CKPT            : チェックポイントの GCS パス
    CONFIG          : UniAD config ファイルパス (コンテナ内)
    N_GPUS          : 使用 GPU 数
    GCS_DATA_ROOT   : nuScenes データの GCS パス
    GCS_CKPT_ROOT   : チェックポイントの GCS パス
"""
import os
import subprocess
import sys
import glob
from pathlib import Path
from datetime import datetime

# ── 環境変数の読み込み ───────────────────────────────────────
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "gs://myad-myad/results/local")
EVAL_TASKS    = os.environ.get("EVAL_TASKS",    "all")
EVAL_MODE     = os.environ.get("EVAL_MODE",     "normal")
CONFIG        = os.environ.get("CONFIG",        "projects/configs/stage2_e2e/base_e2e.py")
N_GPUS        = int(os.environ.get("N_GPUS",   "1"))
GCS_DATA_ROOT = os.environ.get("GCS_DATA_ROOT", "")
GCS_CKPT_ROOT = os.environ.get("GCS_CKPT_ROOT", "")
CKPT_GCS      = os.environ.get("CKPT", f"{GCS_CKPT_ROOT}/uniad_base_e2e.pth")

# ローカル作業ディレクトリ
WORK_DIR      = Path("/workspace/UniAD")
LOCAL_CKPT    = Path("/tmp/ckpts/model.pth")
LOCAL_RESULTS = Path("/tmp/eval_results")
TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], **kwargs) -> None:
    """コマンドを実行してエラーがあれば例外を投げる。"""
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def gsutil_cp(src: str, dst: str, recursive: bool = False) -> None:
    """GCS との間でファイルをコピーする。"""
    cmd = ["gsutil"]
    if recursive:
        cmd += ["-m", "cp", "-r"]
    else:
        cmd += ["cp"]
    cmd += [src, dst]
    run(cmd)


def download_from_gcs() -> None:
    """GCS から nuScenes データとチェックポイントをローカルに取得。"""
    print("=" * 50)
    print(" GCS からデータをダウンロード中...")
    print("=" * 50)

    # チェックポイント
    LOCAL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if CKPT_GCS.startswith("gs://"):
        gsutil_cp(CKPT_GCS, str(LOCAL_CKPT))
        print(f"[OK] ckpt: {CKPT_GCS} → {LOCAL_CKPT}")
    else:
        print(f"[INFO] ckpt はローカルパスを使用: {CKPT_GCS}")

    # nuScenes info pkl (軽量、必須)
    info_dst = WORK_DIR / "data/infos"
    info_dst.mkdir(parents=True, exist_ok=True)
    if GCS_DATA_ROOT:
        gsutil_cp(
            f"{GCS_DATA_ROOT}/infos/",
            str(info_dst) + "/",
            recursive=True,
        )
        gsutil_cp(
            f"{GCS_DATA_ROOT}/others/",
            str(WORK_DIR / "data/others") + "/",
            recursive=True,
        )

    # nuScenes 画像データ (大容量: マウント or 事前コピー推奨)
    nuscenes_dst = WORK_DIR / "data/nuscenes"
    if not nuscenes_dst.exists() and GCS_DATA_ROOT:
        print("[INFO] nuScenes データをダウンロード中 (時間がかかります)...")
        nuscenes_dst.mkdir(parents=True, exist_ok=True)
        gsutil_cp(
            f"{GCS_DATA_ROOT}/nuscenes/",
            str(nuscenes_dst) + "/",
            recursive=True,
        )


def run_evaluation() -> Path:
    """評価スクリプトを実行して結果ディレクトリを返す。"""
    out_dir = LOCAL_RESULTS / f"run_{TIMESTAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(LOCAL_CKPT) if LOCAL_CKPT.exists() else CKPT_GCS

    print("=" * 50)
    print(f" 評価開始: {EVAL_TASKS} ({EVAL_MODE})")
    print("=" * 50)

    if EVAL_MODE == "corruption":
        # 悪天候ロバスト性評価
        run([
            "bash", "/workspace/eval_scripts/eval_corruption.sh",
            "--config",  CONFIG,
            "--ckpt",    ckpt_path,
            "--gpus",    str(N_GPUS),
            "--tasks",   EVAL_TASKS,
            "--out-dir", str(out_dir),
        ], cwd=str(WORK_DIR))
    else:
        # 通常評価
        run([
            "bash", "/workspace/eval_scripts/eval_all.sh",
            "--config",  CONFIG,
            "--ckpt",    ckpt_path,
            "--gpus",    str(N_GPUS),
            "--tasks",   EVAL_TASKS,
            "--out-dir", str(out_dir),
        ], cwd=str(WORK_DIR))

    return out_dir


def upload_results(result_dir: Path) -> None:
    """評価結果を GCS にアップロード。"""
    print("=" * 50)
    print(f" 結果を GCS にアップロード: {AIP_MODEL_DIR}")
    print("=" * 50)

    gcs_dst = f"{AIP_MODEL_DIR}/"

    # 全結果をまとめてアップロード
    gsutil_cp(str(result_dir) + "/", gcs_dst, recursive=True)

    # サマリーを標準出力にも表示
    summary_md = result_dir / "summary" / "report.md"
    if summary_md.exists():
        print("\n" + "=" * 50)
        print(" 評価サマリー")
        print("=" * 50)
        print(summary_md.read_text())

    metrics_json = result_dir / "summary" / "metrics.json"
    if metrics_json.exists():
        import json
        metrics = json.loads(metrics_json.read_text())
        print("\n[metrics.json]")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\n全結果を {gcs_dst} に保存しました。")


if __name__ == "__main__":
    try:
        # 1. データ取得
        download_from_gcs()

        # 2. 評価実行
        result_dir = run_evaluation()

        # 3. GCS アップロード
        upload_results(result_dir)

        print("\n[OK] 全処理が完了しました。")
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] コマンド失敗: {e.cmd}")
        print(f"        終了コード: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
