from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE = (96, 96, 3)
DEFAULT_N_CLASSES = 7
DEFAULT_FILTERS = (32, 64, 128)
DEFAULT_DROPOUT = 0.4


def conv_block(x, filters: int):
    for _ in range(2):
        x = layers.Conv2D(
            filters, 3, strides=1, padding="same", kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return layers.MaxPooling2D(pool_size=2, strides=2)(x)


def build_cnn(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    n_classes: int = DEFAULT_N_CLASSES,
    filters: tuple[int, ...] = DEFAULT_FILTERS,
    dropout: float = DEFAULT_DROPOUT,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    for f in filters:
        x = conv_block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)
