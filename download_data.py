"""
download_data.py
----------------
Resilient download script for the Fruits & Vegetables dataset from Kaggle
using kagglehub with automatic resume and retry on network interruption.

Usage:
    python download_data.py
"""

import sys
import time
from pathlib import Path
import yaml
import kagglehub

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    max_retries = 30
    retry_delay = 5
    path = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Downloading/Resuming dataset from Kaggle...")
            path = kagglehub.dataset_download(
                "kritikseth/fruit-and-vegetable-image-recognition"
            )
            print(f"Dataset downloaded successfully to: {path}")
            break
        except Exception as e:
            print(f"Download interrupted ({type(e).__name__}: {e})")
            if attempt < max_retries:
                print(f"Reconnecting and resuming from previous byte in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise

    if path is None:
        raise RuntimeError("Failed to download dataset after maximum retries.")

    # Move splits into the local data/ folder if not already there
    dest_data_dir = Path("data")
    dest_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(path)

    import shutil
    for split in ("train", "validation", "test"):
        src_split = dataset_root / split
        dest_split = dest_data_dir / split
        if src_split.exists() and not dest_split.exists():
            print(f"Moving {split} split to {dest_split}...")
            shutil.move(str(src_split), str(dest_split))

    # Verify discovered splits in local data/ folder
    for split in ("train", "validation", "test"):
        split_dir = dest_data_dir / split
        if split_dir.exists():
            n_classes = sum(1 for d in split_dir.iterdir() if d.is_dir())
            n_images = sum(1 for _ in split_dir.rglob("*") if _.is_file())
            print(f"  {split:12s} -> {n_classes} classes, {n_images} images")

    # Update train_config.yaml with clean relative paths
    cfg_path = Path("train_config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["data"]["train_dir"] = "data/train"
    cfg["data"]["val_dir"] = "data/validation"
    cfg["data"]["test_dir"] = "data/test"
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    print("\ntrain_config.yaml updated with local data/ paths.")


if __name__ == "__main__":
    main()
