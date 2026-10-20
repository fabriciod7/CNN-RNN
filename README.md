## Descripción

- **CNN** — clasificación de lesiones cutáneas sobre el dataset HAM10000 (7 clases), con partición por grupos (`lesion_id`) para evitar data leakage y validación cruzada estratificada (`StratifiedGroupKFold`).
- **RNN (LSTM/GRU)** — pronóstico de temperatura a +24 h sobre el dataset Jena Climate, con split cronológico estricto y comparación contra baselines.

El documento de diseño detallado se encuentra en [`PIPELINE_TP_KERAS_CNN_RNN.md`](PIPELINE_TP_KERAS_CNN_RNN.md).

---

## Estructura de carpetas

```
tp-keras-cnn-rnn/
├── cnn/
│   ├── __init__.py
│   ├── data.py          # Carga, partición por grupos, class weights
│   ├── model.py         # Arquitectura CNN con augmentation
│   ├── train.py         # Entrenamiento por fold
│   └── evaluate.py      # Métricas, curvas, experimento de capacidad
├── rnn/
│   ├── __init__.py
│   ├── data.py          # Limpieza, split cronológico, ventanas
│   ├── model.py         # LSTM y GRU
│   ├── train.py         # Entrenamiento con callbacks
│   └── evaluate.py      # MAE/RMSE, comparación con baselines
├── models/              # Modelos exportados (.keras) y archivos de normalización (.json)
├── data/                # Datasets (no incluidos en el repo, ver abajo)
├── reports/
│   ├── figures/         # Gráficos generados durante la evaluación
│   └── informe_tecnico.md
├── predict.py           # Script de inferencia CLI
├── requirements.txt
└── PIPELINE_TP_KERAS_CNN_RNN.md
```

---

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
pillow
```

Se recomienda usar un entorno virtual o Google Colab con GPU.

---

## Datasets

### CNN — HAM10000 (Skin Cancer MNIST)

1. Descargar desde Kaggle: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Colocar los archivos en:

```
data/ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   # carpeta con imágenes .jpg
└── HAM10000_images_part_2/   # carpeta con imágenes .jpg
```

### RNN — Jena Climate

1. Descargar desde Kaggle: [Jena Climate Dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate) o desde el repositorio oficial de TensorFlow/Keras.
2. Colocar el archivo en:

```
data/jena_climate_2009_2016.csv
```

---

## Entrenamiento

### Módulo CNN

```bash
python -m cnn.train
```

Entrena 5 folds con `StratifiedGroupKFold`. Los modelos se guardan en `models/cnn_fold{N}.keras`. Al finalizar, el mejor fold se exporta como `models/cnn_ham10000.keras` junto con `models/cnn_classes.json`.

### Módulo RNN

```bash
python -m rnn.train
```

Entrena LSTM y GRU con split cronológico 70/20/10. Los modelos se guardan en `models/rnn_jena_lstm.keras` y `models/rnn_jena_gru.keras`. Los parámetros de normalización se guardan en `models/rnn_norm.json`.

---

## Inferencia con `predict.py`

### Clasificación de imagen (CNN)

```bash
python predict.py image --path ruta/a/imagen.jpg
```

Opciones:

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--path` | (requerido) | Ruta a la imagen de entrada |
| `--model-path` | `models/cnn_ham10000.keras` | Modelo Keras exportado |
| `--classes-path` | `models/cnn_classes.json` | JSON de índice → etiqueta |
| `--size` | `96` | Dimensión de resize (cuadrado) |

Salida esperada:

```
Predicted class : mel
Confidence      : 0.8731
```

### Pronóstico de temperatura (RNN)

```bash
python predict.py temperature --window-csv ventana.csv
```

`ventana.csv` debe tener 72 filas (72 horas) y una columna por variable del dataset, sin índice adicional.

Opciones:

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--window-csv` | (requerido) | CSV con la ventana de 72 h |
| `--model-path` | `models/rnn_jena_lstm.keras` | Modelo Keras exportado |
| `--norm-path` | `models/rnn_norm.json` | JSON con media y desvío de train |
| `--target` | `T (degC)` | Columna objetivo |

Salida esperada:

```
Predicted T (degC) in 24 h: 12.54
```

---

## Documento de diseño

Ver [`PIPELINE_TP_KERAS_CNN_RNN.md`](PIPELINE_TP_KERAS_CNN_RNN.md) para la descripción completa de la arquitectura, justificación matemática, protocolo de evaluación y checklist de entrega.
