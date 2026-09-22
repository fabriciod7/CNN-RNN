# Informe Técnico — TP Integrador Redes Neuronales

**Materia:** Inteligencia Artificial — Lic. en Sistemas de Información (FCyT–UADER)  
**Autores:** [COMPLETAR: nombre/s]  
**Fecha de entrega:** 30 de septiembre  

---

## 1. Introducción y datasets

### 1.1 Objetivo

Este trabajo implementa dos módulos de aprendizaje profundo con Keras:

- **CNN** para clasificación de lesiones cutáneas (dataset HAM10000, 7 clases).
- **RNN (LSTM/GRU)** para pronóstico de temperatura a +24 h (dataset Jena Climate, 14 variables).

Ambos módulos siguen un protocolo estricto de partición de datos para evitar data leakage.

### 1.2 Dataset CNN — HAM10000

- **Fuente:** Kaggle "Skin Cancer MNIST: HAM10000".
- **Tamaño:** [COMPLETAR: N imágenes], 7 clases diagnósticas (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
- **Particularidad:** múltiples imágenes por lesión (`lesion_id`). Sin agrupación por lesión, el modelo puede memorizar lesiones repetidas entre train y validación (data leakage).
- **Preprocesamiento:** resize a 96×96, normalización a [0, 1].

### 1.3 Dataset RNN — Jena Climate

- **Fuente:** `jena_climate_2009_2016.csv` (TensorFlow / Kaggle).
- **Tamaño:** [COMPLETAR: N registros] con frecuencia de 10 min, resampleados a horario.
- **Variables:** 14 columnas meteorológicas. Objetivo: `T (degC)`.
- **Tarea:** predecir la temperatura a +24 h usando una ventana de entrada de 72 h.

---

## 2. Módulo CNN

### 2.1 Arquitectura

La red sigue un esquema de bloques convolucionales apilados con pooling, seguidos de Global Average Pooling y una capa densa final.

```
Input (96×96×3)
├── RandomFlip / RandomRotation / RandomZoom  [augmentation, solo en entrenamiento]
├── ConvBlock(32):  Conv2D×2 → BN → ReLU → MaxPool2D  → (48×48×32)
├── ConvBlock(64):  Conv2D×2 → BN → ReLU → MaxPool2D  → (24×24×64)
├── ConvBlock(128): Conv2D×2 → BN → ReLU → MaxPool2D  → (12×12×128)
├── GlobalAveragePooling2D                             → (128,)
├── Dropout(0.4)
└── Dense(7, softmax)                                  → (7,)
```

#### Tabla de dimensiones y parámetros por capa

| Capa | Entrada | Salida | Parámetros |
|------|---------|--------|------------|
| Conv2D(32, 3×3) #1 | 96×96×3 | 96×96×32 | 3·3·3·32 + 32 = **896** |
| Conv2D(32, 3×3) #2 | 96×96×32 | 96×96×32 | 3·3·32·32 + 32 = **9 248** |
| MaxPool2D(2×2) | 96×96×32 | 48×48×32 | 0 |
| Conv2D(64, 3×3) #1 | 48×48×32 | 48×48×64 | 3·3·32·64 + 64 = **18 496** |
| Conv2D(64, 3×3) #2 | 48×48×64 | 48×48×64 | 3·3·64·64 + 64 = **36 928** |
| MaxPool2D(2×2) | 48×48×64 | 24×24×64 | 0 |
| Conv2D(128, 3×3) #1 | 24×24×64 | 24×24×128 | 3·3·64·128 + 128 = **73 856** |
| Conv2D(128, 3×3) #2 | 24×24×128 | 24×24×128 | 3·3·128·128 + 128 = **147 584** |
| MaxPool2D(2×2) | 24×24×128 | 12×12×128 | 0 |
| GlobalAvgPool2D | 12×12×128 | 128 | 0 |
| Dropout(0.4) | 128 | 128 | 0 |
| Dense(7, softmax) | 128 | 7 | 128·7 + 7 = **903** |
| **Total** | | | **~288 000** |

### 2.2 Justificación matemática

**Tamaño de salida de una convolución:**

```
o = floor((i + 2p - k) / s) + 1
```

Con `padding="same"`, `k=3`, `s=1`: `o = i`. Con MaxPool `k=2`, `s=2`: `o = i/2`.  
La resolución decrece: 96 → 48 → 24 → 12.

**Parámetros de una capa Conv2D:** `k · k · C_in · C_out + C_out`  
Versus una capa densa equivalente sobre el mapa completo: `(12·12·128)·7 = 129 024` —la CNN es órdenes de magnitud más eficiente.

**Localidad y pesos compartidos:** el mismo filtro recorre toda la imagen, capturando patrones locales independientemente de su posición.

**Campo receptivo:** [COMPLETAR: calcular campo receptivo efectivo de la última capa conv].

**Equivarianza traslacional:** `Conv2D` es equivariante a traslaciones. `MaxPool` agrega invarianza local: una traslación pequeña no cambia la respuesta del pooling.

**Estabilidad del gradiente:**
- `ReLU`: derivada 1 en la zona activa (no satura hacia arriba), evita la desaparición del gradiente.
- Inicialización He: `Var(W) = 2/n_in`, calibrada para ReLU.
- `BatchNormalization`: centra y escala las activaciones, estabiliza el entrenamiento.

**Regularización:**
- `Dropout(0.4)`: apaga neuronas aleatoriamente, fuerza representaciones redundantes.
- Augmentation (flip, rotation, zoom): expande el espacio de entrenamiento efectivo.
- `GlobalAveragePooling2D` en lugar de `Flatten`: reduce drásticamente los parámetros de la cabeza.
- `EarlyStopping(patience=8)`: detiene el entrenamiento cuando la pérdida de validación deja de mejorar.

**Salida:** `softmax` + `sparse_categorical_crossentropy` equivale a maximizar la verosimilitud logarítmica de las etiquetas.

### 2.3 Partición por grupos y prevención de data leakage

El dataset HAM10000 contiene varias imágenes de la misma lesión (`lesion_id`). Una partición aleatoria puede asignar imágenes de la misma lesión a train y validación simultáneamente: el modelo aprende a reconocer la lesión y no la clase, produciendo métricas artificialmente altas.

**Solución:**
1. `GroupShuffleSplit(test_size=0.15)` separa el holdout de test garantizando que ningún `lesion_id` aparezca en ambos conjuntos.
2. `StratifiedGroupKFold(n_splits=5)` aplica la misma garantía en cada fold de validación cruzada.
3. Un `assert set(groups_train).isdisjoint(groups_val)` verifica la disjunción en tiempo de ejecución.

### 2.4 Tabla de hiperparámetros CNN

| Hiperparámetro | Valor |
|----------------|-------|
| Optimizador | Adam |
| Learning rate inicial | 1e-3 |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=4) |
| Batch size | [COMPLETAR] |
| Épocas máximas | 60 |
| Épocas efectivas (media) | [COMPLETAR] |
| Dropout | 0.4 |
| Filtros | (32, 64, 128) |
| Activación oculta | ReLU |
| Activación salida | softmax |
| Loss | sparse\_categorical\_crossentropy |
| Augmentation | RandomFlip, RandomRotation(0.1), RandomZoom(0.1) |

### 2.5 Métricas por fold

| Fold | Balanced Acc. | F1 macro | ROC-AUC (OvR) | Recall mel |
|------|--------------|----------|---------------|------------|
| 1 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| 2 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| 3 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| 4 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| 5 | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| **Media ± σ** | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

Curvas de aprendizaje (loss train vs val por fold): `reports/figures/cnn_loss_curves.png`  
Matriz de confusión (test final): `reports/figures/cnn_confusion_matrix_test.png`

### 2.6 Experimento de capacidad (sesgo–varianza)

| Modelo | Filtros | Dropout | Train F1 | Val F1 |
|--------|---------|---------|----------|--------|
| Chico | (16, 32) | 0.4 | [COMPLETAR] | [COMPLETAR] |
| Base | (32, 64, 128) | 0.4 | [COMPLETAR] | [COMPLETAR] |
| Grande | (64, 128, 256) | 0.0 | [COMPLETAR] | [COMPLETAR] |

Gráfico: `reports/figures/cnn_capacity_experiment.png`

### 2.7 Resultado en test final (holdout por grupos)

| Métrica | Valor |
|---------|-------|
| Balanced accuracy | [COMPLETAR] |
| F1 macro | [COMPLETAR] |
| ROC-AUC (OvR) | [COMPLETAR] |
| Recall mel | [COMPLETAR] |

---

## 3. Módulo RNN

### 3.1 Justificación LSTM/GRU desde BPTT

**Problema de las RNN simples:** el gradiente hacia atrás en el tiempo es un producto de jacobianos:

```
∂L/∂h_0 = ∏_{t=1}^{T} W_h^T · diag(tanh'(h_{t-1}))
```

- Radio espectral de `W_h` < 1 → el producto tiende a 0 (*desaparición del gradiente*): la red olvida dependencias largas.
- Radio espectral de `W_h` > 1 → el producto explota (*explosión del gradiente*): el entrenamiento diverge.

**LSTM — carrusel de error constante:**  
La celda de memoria `c_t` se actualiza de forma aditiva:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

La derivada `∂c_t/∂c_{t-1} ≈ f_t`. Cuando la compuerta de olvido `f_t ≈ 1`, el gradiente fluye sin atenuarse a través del tiempo. Esto resuelve la desaparición para dependencias largas.

**GRU:** mecanismo equivalente con compuertas de actualización (`z_t`) y reset (`r_t`). Menos parámetros que LSTM, entrenamiento más rápido, rendimiento comparable en muchas tareas.

**Activaciones:**
- Compuertas (`f_t`, `i_t`, `o_t`, `z_t`, `r_t`): sigmoide → rango [0, 1], interpretable como proporción que "pasa".
- Estado/candidato (`c̃_t`, `h_t`): tanh → rango [−1, 1], centrada en 0.
- Salida: lineal (regresión de un solo valor).

**Explosión:** `clipnorm=1.0` reescala el gradiente cuando su norma supera el umbral, sin alterar su dirección.

**Regularización:** dropout en la entrada de la celda (misma máscara en todos los pasos de tiempo, implementación recurrente de Keras) + `EarlyStopping(patience=5)`.

### 3.2 Split cronológico y normalización

El dataset de series de tiempo exige un split **estrictamente cronológico** (70 % train / 20 % validación / 10 % test) sin ningún shuffle. Mezclar datos del futuro en el entrenamiento produce métricas irreales (look-ahead bias).

La normalización (media y desvío estándar) se ajusta **exclusivamente sobre el conjunto de train**. Aplicar la normalización de train a validación y test simula el escenario real de inferencia, donde los estadísticos del futuro son desconocidos.

Las ventanas de entrada (72 pasos) y sus targets (posición `input_width + horizon - 1`) se generan por separado en cada split para que ninguna ventana cruce la frontera train/val o val/test.

Los parámetros de normalización se guardan en `models/rnn_norm.json` para usarse en `predict.py` sin reentrenar.

### 3.3 Tabla de hiperparámetros RNN

| Hiperparámetro | Valor |
|----------------|-------|
| Optimizador | Adam + clipnorm=1.0 |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Épocas máximas | 40 |
| Épocas efectivas LSTM | [COMPLETAR] |
| Épocas efectivas GRU | [COMPLETAR] |
| Dropout | 0.2 |
| Unidades | (64, 32) |
| Ventana de entrada | 72 h |
| Horizonte de predicción | 24 h |
| Activación salida | lineal |
| Loss | MSE |

### 3.4 Resultados contra baselines

| Modelo | MAE (°C) | RMSE (°C) | Parámetros | Tiempo entrenamiento |
|--------|----------|-----------|------------|----------------------|
| Persistencia | [COMPLETAR] | [COMPLETAR] | 0 | — |
| Denso (Flatten → Dense(64) → Dense(1)) | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| LSTM (64, 32) | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |
| GRU (64, 32) | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] | [COMPLETAR] |

Curvas de aprendizaje: `reports/figures/rnn_lstm_loss.png`, `reports/figures/rnn_gru_loss.png`  
Predicción vs real (tramo de test): `reports/figures/rnn_lstm_predictions.png`, `reports/figures/rnn_gru_predictions.png`  
Error por mes: `reports/figures/rnn_lstm_monthly_mae.png`, `reports/figures/rnn_gru_monthly_mae.png`  
Histograma de residuos: `reports/figures/rnn_lstm_residuals.png`, `reports/figures/rnn_gru_residuals.png`

---

## 4. Análisis de underfitting / overfitting y conclusiones

### 4.1 Diagnóstico CNN

[COMPLETAR con análisis de las curvas de aprendizaje por fold]

- **Underfitting (sesgo):** se manifiesta como loss de train alta y val alta de forma pareja. En el experimento de capacidad, el modelo chico (`filters=(16, 32)`) debería mostrar este comportamiento.
- **Overfitting (varianza):** se manifiesta como loss de train baja y val creciente (gap entre curvas). El modelo grande (`filters=(64, 128, 256)`, `dropout=0.0`) debería mostrarlo.
- **Balance:** el modelo base con augmentation y dropout debe situarse entre ambos extremos.

### 4.2 Diagnóstico RNN

[COMPLETAR con análisis de las curvas de aprendizaje de LSTM y GRU]

- La mejora sobre la persistencia y el modelo denso indica que la red captura dependencias temporales de largo plazo.
- [COMPLETAR: análisis de estabilidad por tramo temporal — ¿el MAE varía entre meses? ¿hay estacionalidad no capturada?]
- [COMPLETAR: análisis del histograma de residuos — ¿hay sesgo sistemático hacia arriba o abajo?]

### 4.3 Conclusiones

[COMPLETAR tras obtener resultados de entrenamiento]

- La partición por grupos en CNN es crítica: [COMPLETAR comparación métrica con/sin agrupación si se realizó el experimento].
- LSTM y GRU superan a la persistencia en [COMPLETAR: X %] de reducción de MAE.
- [COMPLETAR: discusión sobre si la ventana de 72 h es suficiente, posibles mejoras con feature engineering].

---

*Este informe fue generado como plantilla base. Completar las secciones marcadas con `[COMPLETAR: ...]` una vez obtenidos los resultados de entrenamiento.*
