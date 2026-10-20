from __future__ import annotations

"""Dataset preparation utilities for the Jena Climate forecasting pipeline.

The process includes reading the raw series, engineering cyclical temporal features,
normalizing the training statistics, and generating fixed-length sliding windows for the RNN.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import keras
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "jena_climate_2009_2016.csv"
DEFAULT_NORM_PATH = REPO_ROOT / "models" / "rnn_norm.json"

DATETIME_COLUMN = "Date Time"
DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S"
TARGET_COLUMN = "T (degC)"
WIND_SPEED_COLUMN = "wv (m/s)"
MAX_WIND_SPEED_COLUMN = "max. wv (m/s)"
WIND_DIRECTION_COLUMN = "wd (deg)"
INVALID_WIND_VALUE = -9999.0

INPUT_WIDTH = 72
HORIZON = 24
BATCH_SIZE = 256
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.2

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_YEAR = 365.2425 * SECONDS_PER_DAY


def load_dataframe(
    data_path: Path | str = DEFAULT_DATA_PATH,
    resample_hourly: bool = True,
) -> pd.DataFrame:
    """Read the raw Jena dataset and keep it sorted by timestamp for time-aware splitting."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Jena Climate dataset not found at {path}. "
            "Place 'jena_climate_2009_2016.csv' inside the 'data/' directory."
        )
    df = pd.read_csv(path)
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], format=DATETIME_FORMAT)
    df = df.sort_values(DATETIME_COLUMN).set_index(DATETIME_COLUMN)
    if resample_hourly:
        df = df.iloc[5::6]
    return clean_wind_values(df)


def clean_wind_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in (WIND_SPEED_COLUMN, MAX_WIND_SPEED_COLUMN):
        if column in out.columns:
            out.loc[out[column] <= INVALID_WIND_VALUE, column] = 0.0
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical seasonal features so the model can learn daily and annual time patterns."""
    out = df.copy()
    wind_speed = out.pop(WIND_SPEED_COLUMN)
    max_wind_speed = out.pop(MAX_WIND_SPEED_COLUMN)
    wind_radians = out.pop(WIND_DIRECTION_COLUMN) * np.pi / 180.0

    out["wx"] = wind_speed * np.cos(wind_radians)
    out["wy"] = wind_speed * np.sin(wind_radians)
    out["max wx"] = max_wind_speed * np.cos(wind_radians)
    out["max wy"] = max_wind_speed * np.sin(wind_radians)

    seconds = out.index.map(pd.Timestamp.timestamp).to_numpy(dtype="float64")
    out["day sin"] = np.sin(seconds * (2 * np.pi / SECONDS_PER_DAY))
    out["day cos"] = np.cos(seconds * (2 * np.pi / SECONDS_PER_DAY))
    out["year sin"] = np.sin(seconds * (2 * np.pi / SECONDS_PER_YEAR))
    out["year cos"] = np.cos(seconds * (2 * np.pi / SECONDS_PER_YEAR))
    return out


def chronological_split(
    df: pd.DataFrame,
    train: float = TRAIN_FRACTION,
    val: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # No shuffling: the validation and test windows must stay in the future of train.
    n = len(df)
    i_train, i_val = int(n * train), int(n * (train + val))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def normalize(
    train_df: pd.DataFrame,
    *others: pd.DataFrame,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> tuple[list[pd.DataFrame], pd.Series, pd.Series]:
    """Standardize features using training-set statistics only to avoid temporal leakage."""
    # Statistics come from train only, otherwise the future leaks into the past.
    if mean is None or std is None:
        mean, std = train_df.mean(), train_df.std()
    return [(d - mean) / std for d in (train_df, *others)], mean, std


def denormalize(
    values: np.ndarray,
    mean: pd.Series | Sequence[float],
    std: pd.Series | Sequence[float],
    target_col: str = TARGET_COLUMN,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    if isinstance(mean, pd.Series) and isinstance(std, pd.Series):
        return np.asarray(values) * std[target_col] + mean[target_col]
    if columns is None:
        raise ValueError("columns is required when mean/std are plain sequences")
    index = list(columns).index(target_col)
    return np.asarray(values) * std[index] + mean[index]


def save_normalization(
    mean: pd.Series,
    std: pd.Series,
    columns: Sequence[str] | None = None,
    path: Path | str = DEFAULT_NORM_PATH,
) -> Path:
    columns = list(columns) if columns is not None else list(mean.index)
    payload = {
        "mean": [float(mean[c]) for c in columns],
        "std": [float(std[c]) for c in columns],
        "columns": columns,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_normalization(
    path: Path | str = DEFAULT_NORM_PATH,
) -> tuple[pd.Series, pd.Series, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = list(payload["columns"])
    mean = pd.Series(payload["mean"], index=columns)
    std = pd.Series(payload["std"], index=columns)
    return mean, std, columns


def make_windows(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
):
    data = df.values.astype("float32")
    targets = df[target_col].values.astype("float32")
    return keras.utils.timeseries_dataset_from_array(
        data=data[:-horizon],
        targets=targets[input_width + horizon - 1 :],
        sequence_length=input_width,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def window_count(n_rows: int, input_width: int = INPUT_WIDTH, horizon: int = HORIZON) -> int:
    return max(0, n_rows - (input_width + horizon - 1))


def target_timestamps(
    df: pd.DataFrame,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
) -> pd.DatetimeIndex:
    return df.index[input_width + horizon - 1 :]


def verify_window_alignment(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
    n_checks: int = 8,
) -> bool:
    dataset = make_windows(
        df,
        target_col=target_col,
        input_width=input_width,
        horizon=horizon,
        batch_size=n_checks,
        shuffle=False,
    )
    inputs, targets = next(iter(dataset))
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    values = df[target_col].values.astype("float32")
    target_index = list(df.columns).index(target_col)
    offset = input_width + horizon - 1

    for i in range(len(targets)):
        assert np.isclose(inputs[i, 0, target_index], values[i]), (
            f"window {i} does not start at row {i}"
        )
        assert np.isclose(targets[i], values[i + offset]), (
            f"target of window {i} is not the value {offset} steps after its start"
        )
    return True


@dataclass
class WindowedDataset:
    train_ds: Any
    val_ds: Any
    test_ds: Any
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    mean: pd.Series
    std: pd.Series
    columns: list[str] = field(default_factory=list)
    target_col: str = TARGET_COLUMN
    input_width: int = INPUT_WIDTH
    horizon: int = HORIZON
    batch_size: int = BATCH_SIZE

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.input_width, len(self.columns))

    @property
    def target_index(self) -> int:
        return self.columns.index(self.target_col)

    @property
    def test_timestamps(self) -> pd.DatetimeIndex:
        return target_timestamps(self.test_df, self.input_width, self.horizon)


def prepare_datasets(
    data_path: Path | str = DEFAULT_DATA_PATH,
    target_col: str = TARGET_COLUMN,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    use_time_features: bool = False,
    shuffle_train: bool = True,
    norm_path: Path | str | None = DEFAULT_NORM_PATH,
    verify: bool = True,
) -> WindowedDataset:
    """Build the chronological train/validation/test splits and convert them to Keras window datasets."""
    df = load_dataframe(data_path)
    if use_time_features:
        df = add_time_features(df)

    train_df, val_df, test_df = chronological_split(df)
    (train_df, val_df, test_df), mean, std = normalize(train_df, val_df, test_df)
    columns = list(df.columns)

    if norm_path is not None:
        save_normalization(mean, std, columns, norm_path)

    if verify:
        verify_window_alignment(train_df, target_col, input_width, horizon)

    splits: Iterable[tuple[pd.DataFrame, bool]] = (
        (train_df, shuffle_train),
        (val_df, False),
        (test_df, False),
    )
    train_ds, val_ds, test_ds = (
        make_windows(split, target_col, input_width, horizon, batch_size, shuffle)
        for split, shuffle in splits
    )

    return WindowedDataset(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        mean=mean,
        std=std,
        columns=columns,
        target_col=target_col,
        input_width=input_width,
        horizon=horizon,
        batch_size=batch_size,
    )
