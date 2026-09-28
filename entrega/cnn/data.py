from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.utils.class_weight import compute_class_weight

DEFAULT_DATA_DIR = Path("data/ham10000")
METADATA_FILE = "HAM10000_metadata.csv"
IMAGE_ID_COLUMN = "image_id"
GROUP_COLUMN = "lesion_id"
LABEL_COLUMN = "dx"
PATH_COLUMN = "image_path"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_IMAGE_SIZE = (96, 96)
SEED = 42


def index_image_paths(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.setdefault(path.stem, path)
    return paths


def load_metadata(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    metadata_path = data_dir / METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Download the HAM10000 dataset into that directory."
        )
    df = pd.read_csv(metadata_path)
    image_paths = index_image_paths(data_dir)
    df[PATH_COLUMN] = df[IMAGE_ID_COLUMN].map(image_paths)
    missing = int(df[PATH_COLUMN].isna().sum())
    if missing:
        raise FileNotFoundError(
            f"{missing} image files listed in {METADATA_FILE} are missing under {data_dir}."
        )
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> tuple[np.ndarray, list[str]]:
    classes = sorted(df[label_column].astype(str).unique())
    lookup = {name: index for index, name in enumerate(classes)}
    labels = df[label_column].astype(str).map(lookup).to_numpy(dtype="int64")
    return labels, classes


def load_images(paths, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    paths = list(paths)
    images = np.empty((len(paths), image_size[0], image_size[1], 3), dtype="float32")
    for position, path in enumerate(paths):
        image = keras.utils.load_img(str(path), target_size=image_size)
        images[position] = keras.utils.img_to_array(image)
    return images / 255.0


def load_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = load_metadata(data_dir)
    labels, classes = encode_labels(df)
    images = load_images(df[PATH_COLUMN].tolist(), image_size)
    groups = df[GROUP_COLUMN].astype(str).to_numpy()
    return images, labels, groups, classes


def split_holdout(X, y, groups, test_size: float = 0.15, seed: int = SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    assert set(groups[dev_idx]).isdisjoint(groups[test_idx])
    return dev_idx, test_idx


def group_folds(X, y, groups, n_splits: int = 5, seed: int = SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(groups[val_idx])
        yield train_idx, val_idx


def random_folds(X, y, n_splits: int = 5, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(X, y)


def fold_class_weight(y_train) -> dict[int, float]:
    # Fitted on the training part of each fold only: using the whole dataset would
    # leak the validation class distribution into the training objective.
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def save_class_index(classes, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index = {str(position): name for position, name in enumerate(classes)}
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path
