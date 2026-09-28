# TP Integrador — Redes Neuronales (Keras CNN–RNN)

**Materia:** Inteligencia Artificial — Lic. en Sistemas de Información (FCyT–UADER)

- **CNN:** clasificación de lesiones cutáneas en 7 clases (HAM10000), con partición por `lesion_id` (`GroupShuffleSplit` + `StratifiedGroupKFold`, k = 5).
- **RNN:** pronóstico de la temperatura a +24 h con una ventana de 72 h (Jena Climate 2009–2016), con time split cronológico 70/20/10. Se comparan LSTM, GRU, un modelo denso y la persistencia.

La justificación y los resultados están en `informe_tecnico.pdf`.

## Estructura

```
cnn/            data.py, model.py, train.py, evaluate.py
rnn/            data.py, model.py, train.py, evaluate.py
models/         modelos exportados (.keras) y metadatos (.json)
reports/        métricas, historiales de entrenamiento y figuras
data/           datasets (no incluidos)
predict.py      inferencia (CNN y RNN)
make_rnn_window.py   genera una ventana de 72 h para la inferencia de la RNN
sample_window.csv    ventana de ejemplo (últimas 72 h del dataset)
```

## Modelos exportados

| Archivo | Descripción |
|---|---|
| `models/cnn_fold1.keras` … `cnn_fold5.keras` | CNN de cada fold; el test y `predict.py` promedian sus probabilidades (*ensemble*) |
| `models/cnn_classes.json` | índice → clase |
| `models/rnn_gru.keras` | GRU (mejor modelo, MAE 2.06 °C) |
| `models/rnn_lstm.keras` | LSTM (MAE 2.11 °C) |
| `models/rnn_dense.keras` | baseline denso (MAE 2.25 °C) |
| `models/rnn_norm.json` | media, desvío y orden de columnas (ajustados solo con train) |

## Instalación

```bash
pip install -r requirements.txt
```

## Datasets

- **HAM10000:** <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000> → `data/ham10000/` (`HAM10000_metadata.csv`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`).
- **Jena Climate:** <https://www.kaggle.com/datasets/mnassrib/jena-climate> → `data/jena_climate_2009_2016.csv`.

Solo son necesarios para reentrenar o reevaluar; la inferencia funciona sin ellos.

## Inferencia

Todos los comandos se ejecutan desde esta carpeta.

```bash
python predict.py image --path ruta/a/imagen.jpg
python predict.py temperature --window-csv sample_window.csv
```

Con `--model-paths` se puede usar un único modelo de fold, y con `--model-path models/rnn_lstm.keras`, la LSTM.

## Entrenamiento y evaluación

```bash
python -m cnn.train
python -m cnn.evaluate --test
python -m rnn.train
python -m rnn.evaluate
python make_rnn_window.py
```
