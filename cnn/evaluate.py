from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cnn.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_IMAGE_SIZE,
    SEED,
    fold_class_weight,
    group_folds,
    load_dataset,
    random_folds,
    split_holdout,
)
from cnn.model import DEFAULT_DROPOUT, DEFAULT_FILTERS, build_cnn  # noqa: E402
from cnn.train import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MODELS_DIR,
    DEFAULT_N_SPLITS,
    DEFAULT_REPORTS_DIR,
)

DEFAULT_FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
SCALAR_METRICS = ("balanced_accuracy", "f1_macro", "roc_auc_ovr")
CAPACITY_CONFIGS = (
    {"name": "small", "filters": (16, 32), "dropout": DEFAULT_DROPOUT},
    {"name": "base", "filters": DEFAULT_FILTERS, "dropout": DEFAULT_DROPOUT},
    {"name": "large", "filters": (64, 128, 256), "dropout": 0.0},
)


def predict_proba(model: keras.Model, X, batch_size: int = 64) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=0)


def compute_metrics(y_true, y_prob, classes) -> dict:
    labels = list(range(len(classes)))
    y_pred = np.asarray(y_prob).argmax(axis=1)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)),
        "recall_per_class": {name: float(value) for name, value in zip(classes, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_metrics(fold_metrics, classes) -> dict:
    aggregated: dict = {}
    for name in SCALAR_METRICS:
        values = [metrics[name] for metrics in fold_metrics]
        aggregated[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    aggregated["recall_per_class"] = {}
    for name in classes:
        values = [metrics["recall_per_class"][name] for metrics in fold_metrics]
        aggregated["recall_per_class"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return aggregated


def load_fold_models(models_dir: Path | str = DEFAULT_MODELS_DIR, n_splits: int = DEFAULT_N_SPLITS):
    models_dir = Path(models_dir)
    models = []
    for fold in range(1, n_splits + 1):
        path = models_dir / f"cnn_fold{fold}.keras"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model {path}. Run cnn/train.py first.")
        models.append(keras.models.load_model(path))
    return models


def load_histories(reports_dir: Path | str = DEFAULT_REPORTS_DIR, n_splits: int = DEFAULT_N_SPLITS):
    reports_dir = Path(reports_dir)
    histories = []
    for fold in range(1, n_splits + 1):
        path = reports_dir / f"cnn_history_fold{fold}.json"
        if path.exists():
            histories.append(json.loads(path.read_text(encoding="utf-8")))
    return histories


def plot_loss_curves(histories, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(histories), figsize=(4 * len(histories), 3.5), squeeze=False)
    for position, history in enumerate(histories):
        axis = axes[0][position]
        axis.plot(history["loss"], label="train")
        axis.plot(history["val_loss"], label="val")
        axis.set_title(f"fold {position + 1}")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(matrix, classes, output_path: Path | str, normalize: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype="float64")
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}" if normalize else f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_capacity_experiment(results, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [result["name"] for result in results]
    parameters = [result["parameters"] for result in results]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(parameters, [result["train_loss"] for result in results], marker="o", label="train")
    axis.plot(parameters, [result["val_loss"] for result in results], marker="o", label="val")
    for position, name in enumerate(names):
        axis.annotate(name, (parameters[position], results[position]["val_loss"]))
    axis.set_xscale("log")
    axis.set_xlabel("trainable parameters")
    axis.set_ylabel("loss")
    axis.set_title("bias-variance vs capacity")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def fit_experiment_model(
    model: keras.Model,
    train_data,
    val_data,
    class_weight,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    X_train, y_train = train_data
    return model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
        verbose=2,
    )


def capacity_experiment(
    X,
    y,
    groups,
    classes,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> list[dict]:
    train_idx, val_idx = next(group_folds(X, y, groups, n_splits=n_splits))
    class_weight = fold_class_weight(y[train_idx])
    input_shape = (image_size[0], image_size[1], 3)
    results = []
    for config in CAPACITY_CONFIGS:
        model = build_cnn(
            input_shape=input_shape,
            n_classes=len(classes),
            filters=tuple(config["filters"]),
            dropout=config["dropout"],
        )
        history = fit_experiment_model(
            model,
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            class_weight,
            epochs=epochs,
            batch_size=batch_size,
        )
        best_epoch = int(np.argmin(history.history["val_loss"]))
        results.append(
            {
                "name": config["name"],
                "filters": list(config["filters"]),
                "dropout": config["dropout"],
                "parameters": int(model.count_params()),
                "train_loss": float(history.history["loss"][best_epoch]),
                "val_loss": float(history.history["val_loss"][best_epoch]),
            }
        )
    plot_capacity_experiment(results, Path(figures_dir) / "cnn_capacity_experiment.png")
    return results


def compare_random_vs_group(
    X,
    y,
    groups,
    classes,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    filters: tuple[int, ...] = DEFAULT_FILTERS,
    dropout: float = DEFAULT_DROPOUT,
) -> dict:
    input_shape = (image_size[0], image_size[1], 3)
    splits = {
        "group": next(group_folds(X, y, groups, n_splits=n_splits)),
        "random": next(random_folds(X, y, n_splits=n_splits)),
    }
    comparison = {}
    for name, (train_idx, val_idx) in splits.items():
        model = build_cnn(
            input_shape=input_shape,
            n_classes=len(classes),
            filters=tuple(filters),
            dropout=dropout,
        )
        fit_experiment_model(
            model,
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            fold_class_weight(y[train_idx]),
            epochs=epochs,
            batch_size=batch_size,
        )
        y_prob = predict_proba(model, X[val_idx], batch_size=batch_size)
        comparison[name] = compute_metrics(y[val_idx], y_prob, classes)
    return comparison


def evaluate_folds(
    models,
    X_dev,
    y_dev,
    groups_dev,
    classes,
    n_splits: int = DEFAULT_N_SPLITS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    fold_metrics = []
    for model, (_, val_idx) in zip(
        models, group_folds(X_dev, y_dev, groups_dev, n_splits=n_splits)
    ):
        y_prob = predict_proba(model, X_dev[val_idx], batch_size=batch_size)
        fold_metrics.append(compute_metrics(y_dev[val_idx], y_prob, classes))
    return fold_metrics


def evaluate_holdout(models, X_test, y_test, classes, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    # Single final evaluation: the fold models are averaged into one ensemble so the
    # test holdout is never used to pick among them.
    probabilities = np.mean(
        [predict_proba(model, X_test, batch_size=batch_size) for model in models], axis=0
    )
    return compute_metrics(y_test, probabilities, classes)


def run_evaluation(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    run_capacity: bool = False,
    run_leakage_check: bool = False,
    run_test: bool = False,
) -> dict:
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    X, y, groups, classes = load_dataset(data_dir, image_size)
    dev_idx, test_idx = split_holdout(X, y, groups)
    X_dev, y_dev, groups_dev = X[dev_idx], y[dev_idx], groups[dev_idx]

    models = load_fold_models(models_dir, n_splits)
    fold_metrics = evaluate_folds(
        models, X_dev, y_dev, groups_dev, classes, n_splits=n_splits, batch_size=batch_size
    )
    report = {
        "classes": classes,
        "folds": fold_metrics,
        "aggregated": aggregate_metrics(fold_metrics, classes),
    }

    histories = load_histories(reports_dir, n_splits)
    if histories:
        plot_loss_curves(histories, figures_dir / "cnn_loss_curves.png")
    total_confusion = np.sum([metrics["confusion_matrix"] for metrics in fold_metrics], axis=0)
    plot_confusion_matrix(total_confusion, classes, figures_dir / "cnn_confusion_matrix.png")

    if run_capacity:
        report["capacity_experiment"] = capacity_experiment(
            X_dev,
            y_dev,
            groups_dev,
            classes,
            image_size=image_size,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            figures_dir=figures_dir,
        )
    if run_leakage_check:
        report["leakage_check"] = compare_random_vs_group(
            X_dev,
            y_dev,
            groups_dev,
            classes,
            image_size=image_size,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
        )
    if run_test:
        test_metrics = evaluate_holdout(models, X[test_idx], y[test_idx], classes, batch_size)
        report["test"] = test_metrics
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            classes,
            figures_dir / "cnn_confusion_matrix_test.png",
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "cnn_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the HAM10000 CNN folds and holdout.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--capacity-experiment", action="store_true")
    parser.add_argument("--leakage-check", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    keras.utils.set_random_seed(SEED)
    args = parse_args(argv)
    report = run_evaluation(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        image_size=tuple(args.image_size),
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_capacity=args.capacity_experiment,
        run_leakage_check=args.leakage_check,
        run_test=args.test,
    )
    for name in SCALAR_METRICS:
        values = report["aggregated"][name]
        print(f"{name}: {values['mean']:.4f} +/- {values['std']:.4f}")
    if "test" in report:
        print(f"test balanced_accuracy: {report['test']['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
