from __future__ import annotations

from typing import Sequence

import keras
from keras import layers

LEARNING_RATE = 1e-3
CLIP_NORM = 1.0


def build_rnn(
    input_shape: tuple[int, int],
    cell: str = "lstm",
    units: Sequence[int] = (64, 32),
    dropout: float = 0.2,
) -> keras.Model:
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    inputs = keras.Input(shape=input_shape)
    x = Cell(units[0], return_sequences=True, dropout=dropout)(inputs)
    x = Cell(units[1], dropout=dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=f"rnn_{cell}")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=CLIP_NORM),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_dense_baseline(
    input_shape: tuple[int, int],
    units: int = 64,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name="dense_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_model(
    name: str,
    input_shape: tuple[int, int],
    units: Sequence[int] = (64, 32),
    dropout: float = 0.2,
) -> keras.Model:
    if name == "dense":
        return build_dense_baseline(input_shape)
    if name in ("lstm", "gru"):
        return build_rnn(input_shape, cell=name, units=units, dropout=dropout)
    raise ValueError(f"unknown model '{name}', expected one of: dense, lstm, gru")
