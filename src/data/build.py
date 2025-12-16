from pathlib import Path
import os
from sklearn.model_selection import train_test_split

def build_working_data(ddata: Path, work_dir: str = "working_data") -> Path:
    """
    Splits the original training data into training and validation sets (90/10) with a fixed random 
    seed and builds the working dataset structure using symbolic links for train, validation, and test sets
    """
    work_data = Path(work_dir)
    work_data.mkdir(parents=True, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    splits = {"train": {}, "val": {}}

    print("Split train/val 90/10...")

    for cls in ["fake", "real"]:
        all_imgs = [p for ext in exts for p in (ddata / "train" / cls).glob(ext)]

        train_imgs, val_imgs = train_test_split(
            all_imgs, test_size=0.1, shuffle=True, random_state=42
        )

        splits["train"][cls] = train_imgs
        splits["val"][cls] = val_imgs

        for split in ["train", "val"]:
            (work_data / split / cls).mkdir(parents=True, exist_ok=True)

    print("Creating symlink train/val...")
    for split in ["train", "val"]:
        for cls in ["fake", "real"]:
            for img in splits[split][cls]:
                link = work_data / split / cls / img.name
                if not link.exists():
                    os.symlink(img, link)

    print("Creating symlink test...")
    for cls in ["fake", "real"]:
        src = ddata / "test" / cls
        dst = work_data / "test" / cls
        dst.mkdir(parents=True, exist_ok=True)

        for img in src.glob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                link = dst / img.name
                if not link.exists():
                    os.symlink(img, link)

    return work_data
