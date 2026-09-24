import json
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/jena_climate_2009_2016.csv")
OUTPUT_PATH = Path("ventana.csv")
NORM_PATH = Path("models/rnn_norm.json")


def main():
    df = pd.read_csv(DATA_PATH)

    # Asegurarse del formato de fecha usado por el dataset
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.sort_values("Date Time").set_index("Date Time")

    # Mismo preprocesamiento que usa el proyecto para entrenar: una medida por hora.
    df = df.iloc[5::6]

    # Mantener el mismo orden y conjunto de columnas que se usaron al entrenar.
    with open(NORM_PATH, encoding="utf-8") as f:
        norm = json.load(f)
    columns = norm["columns"]

    # La fecha ya no debe quedar en el CSV de entrada del modelo.
    window = df[columns].tail(72).copy()
    if len(window) != 72:
        raise ValueError(f"Se generaron {len(window)} filas; se requieren exactamente 72.")

    window.to_csv(OUTPUT_PATH, index=False)

    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Shape: {window.shape}")
    print(window.head())


if __name__ == "__main__":
    main()
