import argparse
import json
import numpy as np
import keras


def predict_image(
    path,
    model_path="models/cnn_ham10000.keras",
    classes_path="models/cnn_classes.json",
    size=96,
):
    model = keras.models.load_model(model_path)
    with open(classes_path) as f:
        classes = json.load(f)
    img = keras.utils.load_img(path, target_size=(size, size))
    x = keras.utils.img_to_array(img)[None] / 255.0
    probs = model.predict(x)[0]
    return classes[str(int(probs.argmax()))], float(probs.max())


def predict_temperature(
    window,
    model_path="models/rnn_jena_lstm.keras",
    norm_path="models/rnn_norm.json",
    target="T (degC)",
):
    model = keras.models.load_model(model_path)
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm["mean"])
    std = np.array(norm["std"])
    x = ((window - mean) / std)[None].astype("float32")
    y = model.predict(x)[0, 0]
    idx = norm["columns"].index(target)
    return float(y * std[idx] + mean[idx])


def _load_window_csv(csv_path):
    import pandas as pd
    return pd.read_csv(csv_path).values.astype("float32")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for CNN (skin lesion) and RNN (temperature) models."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    img_parser = subparsers.add_parser("image", help="Classify a skin lesion image.")
    img_parser.add_argument("--path", required=True, help="Path to input image.")
    img_parser.add_argument(
        "--model-path", default="models/cnn_ham10000.keras", help="Path to Keras model."
    )
    img_parser.add_argument(
        "--classes-path",
        default="models/cnn_classes.json",
        help="Path to class index JSON.",
    )
    img_parser.add_argument(
        "--size", type=int, default=96, help="Image resize dimension (square)."
    )

    temp_parser = subparsers.add_parser(
        "temperature", help="Predict temperature 24h ahead."
    )
    temp_parser.add_argument(
        "--window-csv",
        required=True,
        help="CSV file with 72 rows x feature columns (no header index).",
    )
    temp_parser.add_argument(
        "--model-path",
        default="models/rnn_jena_lstm.keras",
        help="Path to Keras model.",
    )
    temp_parser.add_argument(
        "--norm-path",
        default="models/rnn_norm.json",
        help="Path to normalization JSON.",
    )
    temp_parser.add_argument(
        "--target",
        default="T (degC)",
        help="Target column name used during training.",
    )

    args = parser.parse_args()

    if args.mode == "image":
        label, confidence = predict_image(
            path=args.path,
            model_path=args.model_path,
            classes_path=args.classes_path,
            size=args.size,
        )
        print(f"Predicted class : {label}")
        print(f"Confidence      : {confidence:.4f}")

    elif args.mode == "temperature":
        window = _load_window_csv(args.window_csv)
        predicted_temp = predict_temperature(
            window=window,
            model_path=args.model_path,
            norm_path=args.norm_path,
            target=args.target,
        )
        print(f"Predicted T (degC) in 24 h: {predicted_temp:.2f}")


if __name__ == "__main__":
    main()
