# Pipeline TP Integrador — Redes Neuronales (Keras CNN + RNN)

**Materia:** Inteligencia Artificial — Lic. en Sistemas de Información (FCyT–UADER)
**Entrega:** 30 de septiembre
**Vía:** email a TODOS los docentes (Hadad, López, Díaz)
**Asunto:** `ENTREGA TP KERAS CNN-RNN REGULARIDAD-APELLIDO1, APELLIDO2`

---

## Cronograma sugerido

| Días | Tarea |
|------|-------|
| 1 | Entorno, datasets, estructura del proyecto |
| 2–4 | Módulo CNN (datos, modelo, Group K-fold, evaluación) |
| 5–7 | Módulo RNN (time split, baselines, LSTM/GRU, evaluación) |
| 7 | Exportación de modelos + `predict.py` |
| 8–9 | Informe técnico (máx. 5 páginas) + revisión + envío |

---

## 0. Setup

### 0.1 Datasets

- **CNN → HAM10000** (Kaggle: "Skin Cancer MNIST: HAM10000")
  - Tiene la columna `lesion_id`, con varias imágenes por lesión. Agrupar por ella evita el data leakage que pide la consigna.
  - Resize a 64×64 o 96×96.
  - Alternativa: Chest X-Ray Pneumonia, agrupando por ID de paciente (se saca del nombre del archivo).
- **RNN → Jena Climate** (`jena_climate_2009_2016.csv`)
  - 14 variables, un registro cada 10 min.
  - Objetivo: predecir `T (degC)` a +24 h usando las últimas 72 h.

### 0.2 Estructura

```
tp-keras-cnn-rnn/
├── cnn/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── rnn/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── reports/figures/
├── predict.py
├── requirements.txt
└── README.md
```

### 0.3 requirements.txt

```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
pillow
```

### 0.4 Reproducibilidad

```python
import keras
keras.utils.set_random_seed(42)
```

### Checklist setup
- [ ] Entorno creado (Colab con GPU o local)
- [ ] Datasets descargados
- [ ] Estructura de carpetas creada
- [ ] Semilla fijada en todos los scripts

> **Regla del TP:** todo el código en inglés (variables, funciones, strings) y sin comentarios innecesarios.

---

## 1. Módulo CNN

### 1.1 `cnn/data.py` — Carga y partición sin fuga

**Pasos:**
1. Leer `HAM10000_metadata.csv`, mapear `image_id` a su ruta y codificar `dx` como etiqueta.
2. Cargar las imágenes, hacer el resize y normalizar a [0,1].
3. Separar el **test final por grupos** (`GroupShuffleSplit`, 15%). No se toca hasta el final.
4. Aplicar **`StratifiedGroupKFold(n_splits=5)`** sobre el resto.
5. Verificar con un assert que ningún grupo se repite entre train y validación.

```python
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

def split_holdout(X, y, groups, test_size=0.15, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    assert set(groups[dev_idx]).isdisjoint(groups[test_idx])
    return dev_idx, test_idx

def group_folds(X, y, groups, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(groups[val_idx])
        yield train_idx, val_idx
```

**Desbalance:** calcular `class_weight` **solo con el train de cada fold**.

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(weights))
```

- [ ] Imágenes cargadas y normalizadas
- [ ] Holdout de test separado por grupos
- [ ] 5 folds con disjunción verificada por assert
- [ ] `class_weight` calculado por fold

### 1.2 `cnn/model.py` — Arquitectura

```python
import keras
from keras import layers

def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return layers.MaxPooling2D(pool_size=2, strides=2)(x)

def build_cnn(input_shape=(96, 96, 3), n_classes=7, filters=(32, 64, 128), dropout=0.4):
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
```

> La augmentation va como capas del modelo: Keras la aplica solo en `training=True`, nunca en validación ni test.

- [ ] `model.summary()` guardado (va al informe)
- [ ] Tabla de dimensiones capa por capa armada

### 1.3 Justificación matemática (para el informe)

- **Tamaño de salida:** `o = floor((i + 2p − k) / s) + 1`
  - Conv 3×3, `padding="same"`, s=1 → conserva la resolución.
  - MaxPool 2×2, s=2 → la divide por 2: 96 → 48 → 24 → 12.
- **Parámetros de una conv:** `k·k·C_in·C_out + C_out`
  - Ejemplo: 3·3·32·64 + 64 = 18.496. Compararlo contra una capa densa equivalente (millones).
- **Localidad espacial y pesos compartidos:** el mismo filtro recorre toda la imagen.
- **Campo receptivo:** calcular cuánto ve una neurona de la última capa.
- **Equivarianza traslacional:** la convolución es equivariante; el pooling agrega invarianza local a desplazamientos pequeños ("equivarianza parcial").
- **Estabilidad del gradiente:** ReLU (derivada 1 en la zona activa, no satura), inicialización He (`Var = 2/n_in`), BatchNorm.
- **Regularización:** Dropout, augmentation, GAP en lugar de Flatten (menos parámetros), EarlyStopping.
- **Salida:** softmax + categorical crossentropy, que equivale a máxima verosimilitud.

### 1.4 `cnn/train.py` — Entrenamiento por fold

```python
def train_fold(model, train_ds, val_ds, class_weight, fold, epochs=60):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        keras.callbacks.ModelCheckpoint(f"models/cnn_fold{fold}.keras", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=callbacks)
    return history
```

- [ ] 5 folds entrenados, `history` guardado (json/pickle)
- [ ] Mejor modelo identificado

### 1.5 `cnn/evaluate.py` — Métricas y sesgo/varianza

**Métricas** (la accuracy sola engaña con clases desbalanceadas):
- Balanced accuracy
- F1 macro
- Recall por clase (clave: melanoma `mel`)
- ROC-AUC (one-vs-rest)
- Matriz de confusión

```python
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

y_prob = model.predict(X_val)
y_pred = y_prob.argmax(axis=1)
metrics = {
    "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
    "f1_macro": f1_score(y_val, y_pred, average="macro"),
    "roc_auc_ovr": roc_auc_score(y_val, y_prob, multi_class="ovr"),
}
```

**Análisis empírico:**
1. Tabla de métricas por fold con **media ± desvío**. Un desvío alto indica varianza.
2. Curvas de loss train vs val por fold:
   - Train alta y val alta, parejas → **underfitting** (sesgo).
   - Train baja y val sube (gap creciente) → **overfitting** (varianza).
3. **Experimento de capacidad:** entrenar 3 modelos.
   - Chico: `filters=(16, 32)`
   - Base: `filters=(32, 64, 128)`
   - Grande: `filters=(64, 128, 256)`, `dropout=0.0`
   - Graficar el error de train y val contra la capacidad (curva sesgo–varianza).
4. **Prueba de fuga (opcional, suma mucho):** comparar split aleatorio vs GroupKFold. Si el aleatorio da mejor resultado, queda demostrado que memorizaba lesiones repetidas.
5. Evaluar **una sola vez** sobre el test final por grupos.

- [ ] Tabla media ± desvío
- [ ] Curvas de aprendizaje en `reports/figures/`
- [ ] Matriz de confusión
- [ ] Experimento de capacidad
- [ ] (Opcional) Comparación aleatorio vs grupos
- [ ] Resultado en test final

---

## 2. Módulo RNN

### 2.1 `rnn/data.py` — Limpieza, time split y ventanas

**Pasos:**
1. Parsear `Date Time`, ordenar cronológicamente y resamplear a horario (`df = df.iloc[5::6]`).
2. Limpiar los valores anómalos: `wv (m/s)` y `max. wv (m/s)` tienen -9999 → reemplazar por 0.
3. (Opcional) Feature engineering: pasar el viento a vector (seno/coseno de `wd (deg)`) y agregar variables cíclicas de hora del día y día del año.
4. **Split cronológico 70/20/10**, sin shuffle nunca.
5. **Normalizar con la media y el desvío de train solamente.**
6. Generar las ventanas por separado en cada split, para que ninguna cruce una frontera.

```python
def chronological_split(df, train=0.7, val=0.2):
    n = len(df)
    i_train, i_val = int(n * train), int(n * (train + val))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]

def normalize(train_df, *others):
    mean, std = train_df.mean(), train_df.std()
    return [(d - mean) / std for d in (train_df, *others)], mean, std

def make_windows(df, target_col, input_width=72, horizon=24, batch_size=256, shuffle=False):
    data = df.values.astype("float32")
    targets = df[target_col].values.astype("float32")
    return keras.utils.timeseries_dataset_from_array(
        data=data[:-horizon],
        targets=targets[input_width + horizon - 1:],
        sequence_length=input_width,
        batch_size=batch_size,
        shuffle=shuffle,
    )
```

> Verificar a mano que el target de la ventana `i` sea exactamente el valor `input_width + horizon − 1` pasos después del inicio.
> El `shuffle=True` en train solo mezcla ventanas ya formadas dentro del pasado; no rompe el orden interno ni el time split.

- [ ] Datos limpios y resampleados
- [ ] Split cronológico sin shuffle
- [ ] Normalización ajustada solo en train (guardar `mean` y `std`)
- [ ] Ventanas verificadas

### 2.2 Baselines (obligatorios para poder comparar)

```python
def persistence_mae(df, target_col, horizon, mean, std):
    y = df[target_col].values
    pred, true = y[:-horizon], y[horizon:]
    return np.mean(np.abs(pred - true)) * std[target_col]
```

- Persistencia: "en 24 h hará la misma temperatura que ahora".
- Modelo denso: `Flatten` → `Dense(64, relu)` → `Dense(1)`.

- [ ] MAE persistencia (°C)
- [ ] MAE denso (°C)

### 2.3 `rnn/model.py` — LSTM y GRU

```python
def build_rnn(input_shape, cell="lstm", units=(64, 32), dropout=0.2):
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    inputs = keras.Input(shape=input_shape)
    x = Cell(units[0], return_sequences=True, dropout=dropout)(inputs)
    x = Cell(units[1], dropout=dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    return model
```

### 2.4 Justificación desde BPTT (para el informe)

- **RNN simple:** el gradiente hacia atrás es un producto de jacobianos `∏ W_hᵀ · diag(tanh'(·))`.
  - Radio espectral de `W_h` < 1 → **desaparición**.
  - Radio espectral de `W_h` > 1 → **explosión**.
- **LSTM:** `c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t`
  - Por la actualización aditiva, `∂c_t/∂c_{t−1} ≈ f_t`. Con `f_t ≈ 1` el gradiente fluye sin atenuarse (carrusel de error constante).
- **GRU:** mecanismo análogo con las compuertas update y reset, menos parámetros y entrenamiento más rápido.
- **Explosión:** `clipnorm=1.0` reescala el gradiente cuando su norma supera el umbral.
- **Activaciones:** sigmoide en las compuertas (rango [0,1], proporción que pasa); tanh en el estado (centrada en 0, acotada).
- **Regularización:** dropout en la entrada de la celda (misma máscara en todos los pasos) + EarlyStopping.

### 2.5 `rnn/train.py`

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("models/rnn_lstm.keras", save_best_only=True),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=callbacks)
```

- [ ] LSTM entrenada
- [ ] GRU entrenada
- [ ] `history` guardado

### 2.6 `rnn/evaluate.py` — Estabilidad predictiva

- MAE y RMSE en **°C** (desnormalizar: `pred * std + mean`).
- Curvas de loss train vs val (diagnóstico under/overfitting).
- Gráfico de predicción vs valor real sobre un tramo del test.
- Error por tramo temporal (por mes): ¿el modelo se degrada?
- Histograma de residuos: ¿hay sesgo sistemático?
- (Opcional) Walk-forward con `TimeSeriesSplit(n_splits=5)` → MAE por corte.

**Tabla comparativa:**

| Modelo | MAE (°C) | RMSE (°C) | Parámetros | Tiempo |
|--------|----------|-----------|------------|--------|
| Persistencia | | | 0 | — |
| Denso | | | | |
| LSTM | | | | |
| GRU | | | | |

- [ ] Tabla completa
- [ ] Gráficos en `reports/figures/`

---

## 3. Exportación e inferencia

```python
model.save("models/cnn_ham10000.keras")
model.save("models/rnn_jena_lstm.keras")
```

Guardar además:
- [ ] `models/cnn_classes.json` (índice → etiqueta)
- [ ] `models/rnn_norm.json` (media y desvío de train por variable)

### `predict.py`

```python
import json, numpy as np, keras

def predict_image(path, model_path="models/cnn_ham10000.keras", classes_path="models/cnn_classes.json", size=96):
    model = keras.models.load_model(model_path)
    classes = json.load(open(classes_path))
    img = keras.utils.load_img(path, target_size=(size, size))
    x = keras.utils.img_to_array(img)[None] / 255.0
    probs = model.predict(x)[0]
    return classes[str(int(probs.argmax()))], float(probs.max())

def predict_temperature(window, model_path="models/rnn_jena_lstm.keras", norm_path="models/rnn_norm.json", target="T (degC)"):
    model = keras.models.load_model(model_path)
    norm = json.load(open(norm_path))
    mean, std = np.array(norm["mean"]), np.array(norm["std"])
    x = ((window - mean) / std)[None].astype("float32")
    y = model.predict(x)[0, 0]
    idx = norm["columns"].index(target)
    return float(y * std[idx] + mean[idx])
```

- [ ] `predict.py` probado con ambos modelos

---

## 4. Informe técnico (máx. 5 páginas)

1. **Introducción y datasets** (½ pág.)
2. **CNN** (2 pág.)
   - Arquitectura: tabla de dimensiones y parámetros
   - Justificación matemática (sección 1.3)
   - Partición por grupos y por qué evita la fuga
   - Tabla de hiperparámetros
   - Métricas por fold (media ± desvío), curvas y experimento de capacidad
3. **RNN** (1½ pág.)
   - Justificación LSTM/GRU desde BPTT (sección 2.4)
   - Time split y normalización solo con train
   - Resultados contra baselines, estabilidad
4. **Análisis under/overfitting y conclusiones** (1 pág.)

**Tabla de hiperparámetros (una por modelo):**

| Hiperparámetro | CNN | RNN |
|----------------|-----|-----|
| Optimizador | Adam | Adam + clipnorm |
| Learning rate | 1e-3 (ReduceLROnPlateau) | 1e-3 |
| Batch size | | 256 |
| Épocas efectivas | | |
| Dropout | 0.4 | 0.2 |
| Activaciones | ReLU / softmax | sigmoid / tanh / lineal |
| Loss | categorical crossentropy | MSE |

- [ ] Informe de 5 páginas o menos
- [ ] Exportado a PDF

---

## 5. Checklist final contra la consigna

- [ ] Conv2D con filtros, padding y stride justificados
- [ ] Capas de pooling (equivarianza traslacional parcial)
- [ ] Group K-fold / partición por lote, sin fuga (con assert)
- [ ] Métricas pertinentes + análisis de sesgo y varianza
- [ ] RNN/LSTM/GRU para pronóstico multivariado
- [ ] Celda justificada desde BPTT (desaparición y explosión)
- [ ] Time split cronológico (el futuro no contamina el pasado)
- [ ] Pérdida secuencial minimizada + estabilidad predictiva analizada
- [ ] Informe: activaciones, hiperparámetros, under/overfitting, 5 páginas o menos
- [ ] Código en inglés, sin comentarios innecesarios
- [ ] Modelos exportados + `predict.py`
- [ ] Email a los 3 docentes con el asunto exacto
- [ ] Adjuntos: PDF + zip de código (modelos por Drive/GitHub si pesan mucho)
