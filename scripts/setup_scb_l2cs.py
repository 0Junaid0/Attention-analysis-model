"""Clone SCB-dataset + L2CS-Net and download the files this pipeline uses.

  SCB:    https://github.com/Whiffe/SCB-dataset.git
  L2CS:   https://github.com/Ahmednull/L2CS-Net.git
  DAiSEE: kagglehub dataset olgaparfenova/daisee

FER2013 and RAF-DB are not used.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party"
DATA = ROOT / "data" / "raw"
CKPT = ROOT / "checkpoints"

SCB_GIT = "https://github.com/Whiffe/SCB-dataset.git"
L2CS_GIT = "https://github.com/Ahmednull/L2CS-Net.git"
YOLOV7_GIT = "https://github.com/WongKinYiu/yolov7.git"

HF_SCB = "wintonYF/SCB-Dataset"
SCB_WEIGHTS_FILE = "SCB5-Handrise-Read-write-2024-9-17/exp/weights/best.pt"
SCB_ZIP_FILE = (
    "SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
)
L2CS_REPO = "py-feat/l2cs"
L2CS_FILE = "l2cs_gaze360_resnet50.safetensors"


def _clone(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        print(f"Already cloned: {dest}")
        subprocess.check_call(["git", "-C", str(dest), "pull", "--ff-only"])
        return
    if dest.exists():
        shutil.rmtree(dest)
    print(f"Cloning {url}")
    subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])


def _hf_download(repo_id: str, filename: str, dest: Path, repo_type: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already have {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    print(f"Downloading {repo_id}/{filename}")
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
        )
        shutil.copy2(cached, dest)
    except Exception:
        from urllib.request import urlretrieve

        kind = "datasets" if repo_type == "dataset" else "models"
        url = f"https://huggingface.co/{kind}/{repo_id}/resolve/main/{filename}"
        print(f"huggingface_hub unavailable; using {url}")
        urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _extract_scb(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "EXTRACTED"
    if marker.exists():
        print(f"SCB already extracted at {out_dir}")
        return
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text(str(zip_path), encoding="utf-8")
    n_img = sum(
        1 for p in out_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Extracted. images~{n_img}")


def _write_data_yaml(scb_root: Path) -> Path:
    """Point Ultralytics at local SCB images/labels."""
    images = None
    for cand in scb_root.rglob("images"):
        if (cand / "train").exists() or any(cand.glob("*.jpg")):
            images = cand
            break
    names = "['hand-raising','reading','writing']"
    if images is None:
        yaml_path = scb_root / "scb.yaml"
        yaml_path.write_text(
            "path: "
            + str(scb_root).replace("\\", "/")
            + f"\ntrain: images/train\nval: images/val\nnc: 3\nnames: {names}\n",
            encoding="utf-8",
        )
        print(f"Wrote template {yaml_path} (fill train/val once images are in place)")
        return yaml_path

    train = "images/train" if (images / "train").exists() else "images"
    val = "images/val" if (images / "val").exists() else train
    root = images.parent
    yaml_path = root / "scb.yaml"
    yaml_path.write_text(
        f"path: {str(root).replace(chr(92), '/')}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        "nc: 3\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")
    return yaml_path


def _daisee_present(dest: Path) -> bool:
    if not dest.exists():
        return False
    if (dest / "DAiSEE" / "Labels").is_dir():
        return True
    for pattern in ("*.avi", "*.mp4"):
        if any(dest.rglob(pattern)):
            return True
    return False


def _download_daisee() -> None:
    """Same entry point as SCB and L2CS: one setup script, skip if already on disk."""
    dest = DATA / "daisee"
    if _daisee_present(dest):
        print(f"Already have DAiSEE at {dest}")
        return
    import importlib.util

    script = ROOT / "scripts" / "download_daisee_kaggle.py"
    spec = importlib.util.spec_from_file_location("download_daisee_kaggle", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = module.download_daisee()
    module.attach_to_project(cache, dest)
    n_files, n_videos = module._count(cache)
    print(f"DAiSEE files: {n_files} ({n_videos} videos)")
    print(f"DAiSEE path:  {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scb-zip", action="store_true")
    parser.add_argument("--skip-l2cs-weights", action="store_true")
    parser.add_argument("--skip-scb-weights", action="store_true")
    parser.add_argument("--skip-daisee", action="store_true", help="Skip the ~14GB DAiSEE download")
    args = parser.parse_args()

    _clone(SCB_GIT, THIRD / "SCB-dataset")
    _clone(L2CS_GIT, THIRD / "L2CS-Net")
    _clone(YOLOV7_GIT, THIRD / "yolov7")

    if not args.skip_l2cs_weights:
        _hf_download(
            L2CS_REPO,
            L2CS_FILE,
            CKPT / "l2cs_gaze360_resnet50.safetensors",
            repo_type="model",
        )

    if not args.skip_scb_weights:
        _hf_download(
            HF_SCB,
            SCB_WEIGHTS_FILE,
            CKPT / "scb_yolo.pt",
            repo_type="dataset",
        )

    if not args.skip_scb_zip:
        zpath = DATA / "scb" / "SCB5-Handrise-Read-write.zip"
        try:
            _hf_download(HF_SCB, SCB_ZIP_FILE, zpath, repo_type="dataset")
            _extract_scb(zpath, DATA / "scb" / "SCB5-Handrise-Read-write")
            _write_data_yaml(DATA / "scb" / "SCB5-Handrise-Read-write")
        except Exception as exc:
            print(f"SCB zip download failed ({exc}).")
            print("Pretrained SCB weights are enough to run inference.")
            print("Copy YOLO images/labels into data/raw/scb/ if you want to retrain.")

    if not args.skip_daisee:
        _download_daisee()

    print("\nDone. Inference uses:")
    print("  checkpoints/l2cs_gaze360_resnet50.safetensors")
    print("  checkpoints/scb_yolo.pt")
    print("  data/raw/daisee")
    print("Optional retrain: python scripts/train_scb_yolo.py")
    print("FER2013 / RAF-DB are not part of this setup.")


if __name__ == "__main__":
    main()
