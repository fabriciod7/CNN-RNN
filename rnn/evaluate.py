from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from rnn.data import (
    BATCH_SIZE,
    DEFAULT_DATA_PATH,
    HORIZON,
    INPUT_WIDTH,
    TARGET_COLUMN,
    WindowedDataset,
    add_time_features,
    denormalize,
    load_dataframe,
    make_windows,
    normalize,
    prepare_datasets,
    window_count,
)
from rnn.model import build_model
from rnn.train import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    MODEL_NAMES,
    SEED,
    history_path,
    model_path,
)

MODEL_LABELS = {
    "persistence": "Persistence",
    "dense": "Dense",
    "lstm": "LSTM",
    "gru": "GRU",
}


def figures_dir(reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> Path:
    path = Path(reports_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_predictions(model: keras.Model, dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.asarray(targets) for _, targets in dataset])
    y_pred = model.predict(dataset, verbose=0).reshape(-1)
    return y_true, y_pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def evaluate_model(
    model: keras.Model,
    dataset,
    mean: pd.Series,
    std: pd.Series,
    target_col: str = TARGET_COLUMN,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_true, y_pred = collect_predictions(model, dataset)
    y_true_c = denormalize(y_true, mean, std, target_col)
    y_pred_c = denormalize(y_pred, mean, std, target_col)
    return regression_metrics(y_true_c, y_pred_c), y_true_c, y_pred_c


def persistence_mae(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    mean: pd.Series,
    std: pd.Series,
) -> float:
    y = df[target_col].values
    pred, true = y[:-horizon], y[horizon:]
    return float(np.mean(np.abs(pred - true)) * std[target_col])


def persistence_metrics(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    horizon: int = HORIZON,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> dict[str, float]:
    y = df[target_col].values
    pred, true = y[:-horizon], y[horizon:]
    residuals = (pred - true) * std[target_col]
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def load_history(name: str, reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> dict | None:
    path = history_path(name, reports_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss_curves(
    history: dict,
    name: str,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> Path:
    curves = history["history"] if "history" in history else history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves["loss"], label="train")
    if "val_loss" in curves:
        ax.plot(curves["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_title(f"Loss curves - {MODEL_LABELS.get(name, name)}")
    ax.legend()
    fig.tight_layout()
    out_path = figures_dir(reports_dir) / f"rnn_{name}_loss.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    name: str,
    n_points: int = 500,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> Path:
    span = slice(0, min(n_points, len(y_true)))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamps[span], y_true[span], label="actual")
    ax.plot(timestamps[span], y_pred[span], label="predicted")
    ax.set_xlabel("time")
    ax.set_ylabel("T (degC)")
    ax.set_title(f"Test predictions vs actual - {MODEL_LABELS.get(name, name)}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = figures_dir(reports_dir) / f"rnn_{name}_predictions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def monthly_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"absolute_error": np.abs(np.asarray(y_pred) - np.asarray(y_true))},
        index=pd.DatetimeIndex(timestamps),
    )
    grouped = frame.groupby(frame.index.to_period("M"))["absolute_error"]
    return pd.DataFrame({"mae": grouped.mean(), "count": grouped.size()})


def plot_monthly_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    name: str,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> Path:
    table = monthly_error(y_true, y_pred, timestamps)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(p) for p in table.index], table["mae"].values)
    ax.set_xlabel("month")
    ax.set_ylabel("MAE (degC)")
    ax.set_title(f"Monthly MAE on test - {MODEL_LABELS.get(name, name)}")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    out_path = figures_dir(reports_dir) / f"rnn_{name}_monthly_mae.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    bins: int = 60,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> Path:
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=bins)
    ax.axvline(0.0, linestyle="--")
    ax.axvline(float(np.mean(residuals)), linestyle=":")
    ax.set_xlabel("residual (degC)")
    ax.set_ylabel("frequency")
    ax.set_title(
        f"Residuals - {MODEL_LABELS.get(name, name)} "
        f"(mean={np.mean(residuals):.2f}, std={np.std(residuals):.2f})"
    )
    fig.tight_layout()
    out_path = figures_dir(reports_dir) / f"rnn_{name}_residuals.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def walk_forward(
    name: str = "lstm",
    data_path: Path | str = DEFAULT_DATA_PATH,
    n_splits: int = 5,
    epochs: int = 10,
    target_col: str = TARGET_COLUMN,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
    batch_size: int = BATCH_SIZE,
    use_time_features: bool = False,
) -> pd.DataFrame:
    df = load_dataframe(data_path)
    if use_time_features:
        df = add_time_features(df)

    rows = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        if window_count(len(val_df), input_width, horizon) <= 0:
            continue
        (train_df, val_df), mean, std = normalize(train_df, val_df)
        train_ds = make_windows(train_df, target_col, input_width, horizon, batch_size, True)
        val_ds = make_windows(val_df, target_col, input_width, horizon, batch_size, False)

        model = build_model(name, (input_width, df.shape[1]))
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
        metrics, _, _ = evaluate_model(model, val_ds, mean, std, target_col)
        rows.append({"fold": fold, "start": val_df.index[0], "end": val_df.index[-1], **metrics})
    return pd.DataFrame(rows)


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, values in results.items():
        rows.append(
            {
                "model": MODEL_LABELS.get(name, name),
                "mae_degc": values.get("mae"),
                "rmse_degc": values.get("rmse"),
                "params": values.get("params", 0),
                "training_time_s": values.get("training_time_s"),
            }
        )
    return pd.DataFrame(rows)


def table_to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def save_comparison_table(
    table: pd.DataFrame,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> tuple[Path, Path]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    csv_path = reports_path / "rnn_comparison.csv"
    md_path = reports_path / "rnn_comparison.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(table_to_markdown(table), encoding="utf-8")
    return csv_path, md_path


def evaluate_all(
    data: WindowedDataset | None = None,
    names: tuple[str, ...] = MODEL_NAMES,
    data_path: Path | str = DEFAULT_DATA_PATH,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    use_time_features: bool = False,
) -> pd.DataFrame:
    keras.utils.set_random_seed(SEED)

    if data is None:
        data = prepare_datasets(
            data_path=data_path,
            use_time_features=use_time_features,
            shuffle_train=False,
            norm_path=None,
            verify=True,
        )

    results: dict[str, dict] = {
        "persistence": {
            **persistence_metrics(
                data.test_df, data.target_col, data.horizon, data.mean, data.std
            ),
            "params": 0,
            "training_time_s": None,
        }
    }

    timestamps = data.test_timestamps
    for name in names:
        checkpoint = model_path(name, models_dir)
        if not checkpoint.exists():
            continue
        model = keras.models.load_model(checkpoint)
        metrics, y_true, y_pred = evaluate_model(
            model, data.test_ds, data.mean, data.std, data.target_col
        )
        history = load_history(name, reports_dir)
        results[name] = {
            **metrics,
            "params": int(model.count_params()),
            "training_time_s": None if history is None else history.get("training_time_s"),
        }
        if history is not None:
            plot_loss_curves(history, name, reports_dir)
        plot_predictions(y_true, y_pred, timestamps, name, reports_dir=reports_dir)
        plot_monthly_error(y_true, y_pred, timestamps, name, reports_dir=reports_dir)
        plot_residual_histogram(y_true, y_pred, name, reports_dir=reports_dir)

    table = comparison_table(results)
    save_comparison_table(table, reports_dir)
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Jena Climate forecasting models")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--time-features", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--walk-forward-cell", choices=("lstm", "gru", "dense"), default="lstm")
    parser.add_argument("--walk-forward-splits", type=int, default=5)
    parser.add_argument("--walk-forward-epochs", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    table = evaluate_all(
        data_path=args.data_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        use_time_features=args.time_features,
    )
    print(table.to_string(index=False))

    if args.walk_forward:
        folds = walk_forward(
            name=args.walk_forward_cell,
            data_path=args.data_path,
            n_splits=args.walk_forward_splits,
            epochs=args.walk_forward_epochs,
            use_time_features=args.time_features,
        )
        folds.to_csv(Path(args.reports_dir) / "rnn_walk_forward.csv", index=False)
        print(folds.to_string(index=False))


if __name__ == "__main__":
    main()
