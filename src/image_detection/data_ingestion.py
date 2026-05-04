from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    train_dir: Path
    val_dir: Path
    test_dir: Path
    root_dir: Path | None = None
    pre_split: bool = True


def _first_existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _class_dirs(path: Path) -> list[Path]:
    return [item for item in path.iterdir() if item.is_dir() and not item.name.startswith(".")]


def _looks_like_image_dataset_dir(path: Path) -> bool:
    # Heuristic: class folder with at least one image inside this folder tree.
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in image_exts:
            return True
    return False


def get_dataset_paths(data_root: str) -> DatasetPaths:
    """
    Validate and return dataset split directories.

    Supported structures:
    1) Pre-split:
        data_root/
          train/{real,fake}
          val/{real,fake} or validation/{real,fake}
          test/{real,fake}

    2) Class-only (auto split in dataloader step):
        data_root/
          real/
          fake/
    """
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        # Backward-compatible fallback for older project layout.
        fallback = root.parent / "img dataset"
        if fallback.exists() and fallback.is_dir():
            print(
                f"[data_ingestion] Dataset root not found: {root}. "
                f"Falling back to: {fallback}"
            )
            root = fallback
        else:
            hint = ""
            if root.parent.exists():
                sibling_dirs = [p.name for p in root.parent.iterdir() if p.is_dir()]
                if sibling_dirs:
                    hint = f" Available directories in {root.parent}: {', '.join(sorted(sibling_dirs))}"
            raise FileNotFoundError(f"Dataset root not found: {root}.{hint}")

    train_dir = _first_existing_dir(root, ("train", "training"))
    val_dir = _first_existing_dir(root, ("val", "validation", "valid"))
    test_dir = _first_existing_dir(root, ("test", "testing"))

    if train_dir and val_dir and test_dir:
        return DatasetPaths(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            pre_split=True,
        )

    # Some datasets are nested one level deeper, e.g. data_root/deepfake_images/train/...
    for child in _class_dirs(root):
        nested_train = _first_existing_dir(child, ("train", "training"))
        nested_val = _first_existing_dir(child, ("val", "validation", "valid"))
        nested_test = _first_existing_dir(child, ("test", "testing"))
        if nested_train and nested_val and nested_test:
            return DatasetPaths(
                train_dir=nested_train,
                val_dir=nested_val,
                test_dir=nested_test,
                root_dir=child,
                pre_split=True,
            )

    class_dirs = [path for path in _class_dirs(root) if _looks_like_image_dataset_dir(path)]
    if len(class_dirs) >= 2:
        return DatasetPaths(
            train_dir=root,
            val_dir=root,
            test_dir=root,
            root_dir=root,
            pre_split=False,
        )

    raise FileNotFoundError(
        "Dataset must contain either train/val/test splits or at least two class folders."
    )
 