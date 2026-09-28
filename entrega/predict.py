import argparse
import json

import keras
import numpy as np
import pandas as pd

CNN_FOLD_MODELS = [f"models/cnn_fold{fold}.keras" for fold in range(1, 6)]


def predict_image(
    path,
    model_paths=CNN_FOLD_MODELS,
    classes_path="models/cnn_classes.json",
    size=96,
):
    with open(classes_path) as f:
        classes = json.load(f)
    img = keras.utils.load_img(path, target_size=(size, size))
    x = keras.utils.img_to_array(img)[None] / 255.0
    probs = np.mean(
        [keras.models.load_model(p).predict(x, verbose=0)[0] for p in model_paths], axis=0
    )
    return classes[str(int(probs.argmax()))], float(probs.max())


def predict_temperature(
    window,
    model_path="models/rnn_gru.keras",
    norm_path="models/rnn_norm.json",
    target="T (degC)",
):
    model = keras.models.load_model(model_path)
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm["mean"])
    std = np.array(norm["std"])
    x = ((window - mean) / std)[None].astype("float32")
    y = model.predict(x, verbose=0)[0, 0]
    idx = norm["columns"].index(target)
    return float(y * std[idx] + mean[idx])


def load_window_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "Date Time" in df.columns:
        df = df.drop(columns=["Date Time"])
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().any().any():
        bad_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
        raise ValueError(f"Window CSV contains non-numeric values in columns: {bad_cols}")
    return numeric_df.values.astype("float32")


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for the CNN (skin lesion) and RNN (temperature) models."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    img_parser = subparsers.add_parser("image", help="Classify a skin lesion image.")
    img_parser.add_argument("--path", required=True, help="Path to the input image.")
    img_parser.add_argument(
        "--model-paths",
        nargs="+",
        default=CNN_FOLD_MODELS,
        help="One or more Keras models; their probabilities are averaged.",
    )
    img_parser.add_argument("--classes-path", default="models/cnn_classes.json")
    img_parser.add_argument("--size", type=int, default=96)

    temp_parser = subparsers.add_parser("temperature", help="Predict the temperature 24 h ahead.")
    temp_parser.add_argument(
        "--window-csv", required=True, help="CSV with 72 hourly rows and the training feature columns."
    )
    temp_parser.add_argument("--model-path", default="models/rnn_gru.keras")
    temp_parser.add_argument("--norm-path", default="models/rnn_norm.json")
    temp_parser.add_argument("--target", default="T (degC)")

    args = parser.parse_args()

    if args.mode == "image":
        label, confidence = predict_image(
            path=args.path,
            model_paths=args.model_paths,
            classes_path=args.classes_path,
            size=args.size,
        )
        print(f"Predicted class : {label}")
        print(f"Confidence      : {confidence:.4f}")
    else:
        predicted_temp = predict_temperature(
            window=load_window_csv(args.window_csv),
            model_path=args.model_path,
            norm_path=args.norm_path,
            target=args.target,
        )
        print(f"Predicted T (degC) in 24 h: {predicted_temp:.2f}")


if __name__ == "__main__":
    main()
