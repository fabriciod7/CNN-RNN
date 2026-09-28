import argparse
from pathlib import Path

from rnn.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_NORM_PATH,
    INPUT_WIDTH,
    add_time_features,
    load_dataframe,
    load_normalization,
)


def build_window(data_path=DEFAULT_DATA_PATH, norm_path=DEFAULT_NORM_PATH, input_width=INPUT_WIDTH):
    df = add_time_features(load_dataframe(data_path))
    columns = load_normalization(norm_path)[2]
    window = df[columns].tail(input_width)
    if len(window) != input_width:
        raise ValueError(f"Got {len(window)} rows, expected {input_width}.")
    return window


def main():
    parser = argparse.ArgumentParser(description="Export the last 72 h of Jena Climate as an inference window.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--norm-path", default=str(DEFAULT_NORM_PATH))
    parser.add_argument("--output", default="sample_window.csv")
    args = parser.parse_args()

    window = build_window(args.data_path, args.norm_path)
    window.to_csv(Path(args.output), index=False)
    print(f"Saved {args.output} with shape {window.shape}")


if __name__ == "__main__":
    main()
