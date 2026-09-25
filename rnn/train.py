from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras

from rnn.data import (
    BATCH_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_NORM_PATH,
    HORIZON,
    INPUT_WIDTH,
    TARGET_COLUMN,
    WindowedDataset,
    prepare_datasets,
)
from rnn.model import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

SEED = 42
EPOCHS = 50
PATIENCE = 8
MODEL_NAMES = ("dense", "lstm", "gru")
EXPORT_ALIASES = {"lstm": "rnn_jena_lstm.keras", "gru": "rnn_jena_gru.keras"}


def model_path(name: str, models_dir: Path | str = DEFAULT_MODELS_DIR) -> Path:
    return Path(models_dir) / f"rnn_{name}.keras"


def history_path(name: str, reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> Path:
    return Path(reports_dir) / f"rnn_{name}_history.json"


def train_model(
    model: keras.Model,
    train_ds,
    val_ds,
    checkpoint_path: Path | str,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> tuple[keras.callbacks.History, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True),
    ]
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history, time.perf_counter() - start


def save_history(
    name: str,
    model: keras.Model,
    history: keras.callbacks.History,
    training_time: float,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> Path:
    out_path = history_path(name, reports_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": name,
        "params": int(model.count_params()),
        "epochs_run": len(history.history.get("loss", [])),
        "training_time_s": float(training_time),
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run(
    names: tuple[str, ...] = MODEL_NAMES,
    data: WindowedDataset | None = None,
    data_path: Path | str = DEFAULT_DATA_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    input_width: int = INPUT_WIDTH,
    horizon: int = HORIZON,
    use_time_features: bool = True,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    norm_path: Path | str = DEFAULT_NORM_PATH,
) -> dict[str, dict]:
    keras.utils.set_random_seed(SEED)

    if data is None:
        data = prepare_datasets(
            data_path=data_path,
            target_col=TARGET_COLUMN,
            input_width=input_width,
            horizon=horizon,
            batch_size=batch_size,
            use_time_features=use_time_features,
            norm_path=norm_path,
        )

    results: dict[str, dict] = {}
    for name in names:
        model = build_model(name, data.input_shape)
        model.summary()
        checkpoint = model_path(name, models_dir)
        history, training_time = train_model(
            model, data.train_ds, data.val_ds, checkpoint, epochs=epochs
        )
        save_history(name, model, history, training_time, reports_dir)
        if name in EXPORT_ALIASES:
            model.save(Path(models_dir) / EXPORT_ALIASES[name])
        results[name] = {
            "params": int(model.count_params()),
            "training_time_s": float(training_time),
            "epochs_run": len(history.history.get("loss", [])),
            "checkpoint": str(checkpoint),
        }
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Jena Climate forecasting models")
    parser.add_argument("--cell", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--norm-path", default=str(DEFAULT_NORM_PATH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--input-width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--time-features", action="store_true", help="Enable cyclical time features for stronger temporal context.")
    parser.add_argument("--no-time-features", action="store_false", dest="time_features")
    parser.set_defaults(time_features=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    names = MODEL_NAMES if args.cell == "all" else (args.cell,)
    results = run(
        names=names,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_width=args.input_width,
        horizon=args.horizon,
        use_time_features=args.time_features,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        norm_path=args.norm_path,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
