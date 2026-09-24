from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import keras

from cnn.data import (
    DEFAULT_DATA_DIR,
    DEFAULT_IMAGE_SIZE,
    SEED,
    fold_class_weight,
    group_folds,
    load_dataset,
    save_class_index,
    split_holdout,
)
from cnn.model import DEFAULT_DROPOUT, DEFAULT_FILTERS, build_cnn

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SPLITS = 5


def train_fold(
    model: keras.Model,
    train_data,
    val_data,
    class_weight,
    fold: int,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            str(models_dir / f"cnn_fold{fold}.keras"), save_best_only=True
        ),
    ]
    X_train, y_train = train_data
    return model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )


def save_history(history, fold: int, reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"cnn_history_fold{fold}.json"
    payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_model_summary(model: keras.Model, reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "cnn_model_summary.txt"
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_cross_validation(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    filters: tuple[int, ...] = DEFAULT_FILTERS,
    dropout: float = DEFAULT_DROPOUT,
) -> dict:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    X, y, groups, classes = load_dataset(data_dir, image_size)
    save_class_index(classes, models_dir / "cnn_classes.json")

    dev_idx, test_idx = split_holdout(X, y, groups)
    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]

    input_shape = (image_size[0], image_size[1], 3)
    summary_saved = False
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        group_folds(X_dev, y_dev, groups_dev, n_splits=n_splits), start=1
    ):
        model = build_cnn(
            input_shape=input_shape,
            n_classes=len(classes),
            filters=tuple(filters),
            dropout=dropout,
        )
        if not summary_saved:
            save_model_summary(model, reports_dir)
            summary_saved = True

        class_weight = fold_class_weight(y_dev[train_idx])
        history = train_fold(
            model,
            (X_dev[train_idx], y_dev[train_idx]),
            (X_dev[val_idx], y_dev[val_idx]),
            class_weight,
            fold,
            epochs=epochs,
            batch_size=batch_size,
            models_dir=models_dir,
        )
        save_history(history, fold, reports_dir)
        fold_results.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "epochs_run": int(len(history.history["loss"])),
                "best_val_loss": float(min(history.history["val_loss"])),
                "class_weight": class_weight,
            }
        )

    best_fold = min(fold_results, key=lambda result: result["best_val_loss"])["fold"]
    shutil.copyfile(
        models_dir / f"cnn_fold{best_fold}.keras", models_dir / "cnn_ham10000.keras"
    )

    summary = {
        "classes": classes,
        "image_size": list(image_size),
        "dev_size": int(len(dev_idx)),
        "test_size": int(len(test_idx)),
        "best_fold": best_fold,
        "folds": fold_results,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cnn_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HAM10000 CNN with grouped cross-validation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--filters", type=int, nargs="+", default=list(DEFAULT_FILTERS))
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    keras.utils.set_random_seed(SEED)
    args = parse_args(argv)
    summary = run_cross_validation(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        image_size=tuple(args.image_size),
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        filters=tuple(args.filters),
        dropout=args.dropout,
    )
    for result in summary["folds"]:
        print(
            f"fold {result['fold']}: epochs={result['epochs_run']} "
            f"best_val_loss={result['best_val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
