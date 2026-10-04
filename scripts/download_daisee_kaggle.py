"""Download DAiSEE with the official KaggleHub snippet, then attach it to the repo.

    import kagglehub
    path = kagglehub.dataset_download("olgaparfenova/daisee")
    print("Path to dataset files:", path)

KaggleHub caches files under your user folder. This script also points
data/raw/daisee at that cache so the rest of the pipeline can find them.

Login once if prompted:
  python -c "import kagglehub; kagglehub.login()"
or put kaggle.json in %USERPROFILE%\\.kaggle\\kaggle.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DAISEE = ROOT / "data" / "raw" / "daisee"
KAGGLE_HANDLE = "olgaparfenova/daisee"
SOURCE_MARKER = "KAGGLE_SOURCE.txt"


def download_daisee() -> Path:
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    path = kagglehub.dataset_download(KAGGLE_HANDLE)
    print("Path to dataset files:", path)
    return Path(path)


def _count(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()]
    videos = [p for p in files if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}]
    return len(files), len(videos)


def attach_to_project(cache_path: Path, dest: Path) -> Path:
    """Make data/raw/daisee point at the KaggleHub cache without a second copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() or dest.is_symlink():
        is_link = dest.is_symlink() or (
            hasattr(dest, "is_junction") and dest.is_junction()
        )
        if is_link:
            dest.unlink()
        elif dest.is_dir():
            leftover = [p for p in dest.iterdir() if p.name != SOURCE_MARKER]
            if not leftover:
                for p in dest.iterdir():
                    p.unlink()
                dest.rmdir()
            else:
                (dest / SOURCE_MARKER).write_text(str(cache_path.resolve()), encoding="utf-8")
                print(f"Kept existing folder {dest}")
                print(f"Kaggle cache is {cache_path}")
                return dest
        else:
            dest.unlink()

    linked = False
    if os.name == "nt":
        # Directory junction does not need admin on Windows.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dest), str(cache_path)],
            capture_output=True,
            text=True,
        )
        linked = result.returncode == 0
        if not linked:
            print(result.stdout)
            print(result.stderr)
    else:
        try:
            dest.symlink_to(cache_path, target_is_directory=True)
            linked = True
        except OSError as exc:
            print(f"Symlink failed ({exc}); writing a path marker instead.")

    if not linked:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / SOURCE_MARKER).write_text(str(cache_path.resolve()), encoding="utf-8")
        print(f"Wrote cache path to {dest / SOURCE_MARKER}")
        print("Use that folder, or copy files from the Kaggle cache if you want a local duplicate.")
        return dest

    print(f"Linked {dest} -> {cache_path}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DAiSEE via kagglehub")
    parser.add_argument("--out", default=str(PROJECT_DAISEE), help="Project folder to attach")
    args = parser.parse_args()

    cache_path = download_daisee()
    dest = attach_to_project(cache_path, Path(args.out))
    n_files, n_videos = _count(cache_path)
    print(f"Dataset files: {n_files} ({n_videos} videos)")
    print(f"Project path: {dest}")
    print("Download finished. This does not train the model.")


if __name__ == "__main__":
    main()
