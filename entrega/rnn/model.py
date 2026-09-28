from __future__ import annotations

from typing import Sequence

import keras
from keras import layers

LEARNING_RATE = 1e-4
CLIP_NORM = 1.0
REGULARIZATION = 1e-4


def build_rnn(
    input_shape: tuple[int, int],
    cell: str = "lstm",
    units: Sequence[int] = (64, 64),
    dropout: float = 0.2,
    recurrent_dropout: float = 0.15,
) -> keras.Model:
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    regularizer = keras.regularizers.l2(REGULARIZATION)
    inputs = keras.Input(shape=input_shape)
    x = Cell(
        units[0],
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
    )(inputs)
    x = Cell(
        units[1],
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
    )(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(x)
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
    units: Sequence[int] = (64, 64),
    dropout: float = 0.2,
    recurrent_dropout: float = 0.15,
) -> keras.Model:
    if name == "dense":
        return build_dense_baseline(input_shape)
    if name in ("lstm", "gru"):
        return build_rnn(
            input_shape,
            cell=name,
            units=units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
    raise ValueError(f"unknown model '{name}', expected one of: dense, lstm, gru")
