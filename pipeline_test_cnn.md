# Pipeline de trabajo para la CNN y la documentación del informe

Este documento describe el paso a paso que deberías seguir para que todo lo requerido por la consigna quede implementado, ejecutado y documentado en el informe técnico.

---

## 1. Objetivo

Construir y evaluar una red neuronal convolucional (CNN) en Keras para clasificación de imágenes, con:

- uso sistemático de filtros 2D, convoluciones, padding y stride;
- capas de pooling para reducción de resolución y equivarianza traslacional parcial;
- partición rigurosa por grupos para evitar data leakage;
- análisis empírico de sesgo y varianza;
- métricas adecuadas para clasificación multiclase;
- documentación completa del pipeline y resultados.

---

## 2. Preparación del dataset

### 2.1 Elegir un dataset apropiado

Usar HAM10000, que tiene imágenes RGB y etiquetas de diagnóstico multiclase.

### 2.2 Organizar la estructura de datos

Guardar los datos como:

```text
data/
└── ham10000/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/
```

### 2.3 Verificar consistencia del metadata

Asegurarse de que existan:

- `image_id`
- `dx` (clase diagnóstica)
- `lesion_id` (identificador de lesión/grupo)

Esto es clave porque la partición por grupos se basa en `lesion_id`.

### 2.4 Cargar y normalizar imágenes

- leer cada imagen desde el path del metadata;
- redimensionar a tamaño fijo, por ejemplo `(96, 96, 3)`;
- convertir a float32;
- escalar valores al rango `[0, 1]`.

Ejemplo conceptual:

```python
images = keras.utils.img_to_array(image) / 255.0
```

---

## 3. Diseñar la CNN

### 3.1 Arquitectura requerida

La red debe incluir explícitamente:

- `Conv2D` con filtros 2D;
- `padding="same"`;
- `strides=1` en las capas internas;
- `BatchNormalization`;
- activación `ReLU`;
- capas `MaxPooling2D` con `pool_size=2` y `strides=2`;
- salida final con `Dense(n_classes, activation="softmax")`.

### 3.2 Estructura recomendada

Un bloque tipo:

```python
x = layers.Conv2D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
```

Repetir este bloque para varios filtros, por ejemplo:

```python
filters = (32, 64, 128)
```

### 3.3 Justificación técnica para la consigna

En el informe debes explicar que:

- los filtros preservan la localidad espacial;
- la convolución de 2D permite detectar patrones locales (bordes, texturas, estructuras);
- el padding mantiene la resolución espacial en cada etapa;
- el pooling reduce la resolución y aporta invariancia traslacional parcial;
- la jerarquía de capas extrae features cada vez más abstractas.

---

## 4. Preparar la partición rigurosa

### 4.1 Evitar fuga de información

Esto es obligatorio. No se puede usar split aleatorio si los grupos están repetidos.

### 4.2 Partición por grupos

Usar:

- `GroupShuffleSplit` para dejar un holdout final;
- `StratifiedGroupKFold` para validación cruzada.

Regla clave:

```python
assert set(groups[train_idx]).isdisjoint(groups[val_idx])
```

Esto garantiza que ninguna lesión repetida esté en train y validation al mismo tiempo.

### 4.3 Importancia para el informe

Debe quedar documentado explícitamente que:

- el dataset tiene varios elementos asociados al mismo `lesion_id`;
- si se mezclan por imagen aleatoria, la red puede memorizar artefactos visuales asociados a esa lesión;
- la partición por grupos permite evaluar la verdadera capacidad de generalización.

---

## 5. Entrenamiento por folds

### 5.1 Configurar validación cruzada

Implementar un procedimiento con 5 folds.

### 5.2 Usar class weights

Como hay desbalance de clases, calcular pesos por fold usando solo el conjunto de entrenamiento del fold.

```python
class_weight = fold_class_weight(y_train)
```

Esto evita contaminar la pérdida con la distribución total del dataset.

### 5.3 Guardar resultados por fold

Para cada fold guardar:

- historial de entrenamiento;
- mejor modelo;
- métricas;
- epoch final;
- evaluación en validación.

### 5.4 Selección del mejor fold

Elegir el mejor modelo según valor mínimo de `val_loss` o según la métrica objetivo.

### 5.5 Guardar el modelo final

Exportar por ejemplo:

- `models/cnn_fold1.keras`
- `models/cnn_fold2.keras`
- ...
- `models/cnn_ham10000.keras`
- `models/cnn_classes.json`

---

## 6. Evaluación de la CNN

### 6.1 Métricas mínimas requeridas

Para clasificación multiclase, usar al menos:

- balanced accuracy
- F1 macro
- ROC-AUC (OvR)
- recall de la clase más importante (por ejemplo `mel`)

Ejemplo:

```python
balanced_accuracy_score(y_true, y_pred)
f1_score(y_true, y_pred, average="macro")
roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr")
```

### 6.2 Tabla por fold

Crear una tabla con los 5 folds:

| Fold | Balanced Acc. | F1 macro | ROC-AUC | Recall mel |
|------|---------------|----------|---------|-----------|
| 1    | ...           | ...      | ...     | ...       |
| 2    | ...           | ...      | ...     | ...       |
| 3    | ...           | ...      | ...     | ...       |
| 4    | ...           | ...      | ...     | ...       |
| 5    | ...           | ...      | ...     | ...       |
| Media ± σ | ... | ... | ... | ... |

### 6.3 Matriz de confusión

Guardar la matriz de confusión sobre el test final para ver qué clases se confunden entre sí.

### 6.4 Curvas de aprendizaje

Guardar gráficos de:

- loss train vs val por fold;
- accuracy train vs val si corresponde.

Esto permite diagnosticar:

- underfitting;
- overfitting;
- entrenamiento estable.

---

## 7. Experimento de capacidad (sesgo–varianza)

### 7.1 Qué debe hacerse

Entrenar al menos 3 configuraciones:

- modelo chico: filtros más pequeños, por ejemplo `(16, 32)`
- modelo base: filtros intermedios, por ejemplo `(32, 64, 128)`
- modelo grande: filtros más grandes, por ejemplo `(64, 128, 256)` y sin dropout o con dropout bajo

### 7.2 Qué analizar

- si el modelo chico tiene train loss alto y val loss alto → underfitting;
- si el modelo grande tiene train loss muy bajo y val loss alto y separándose → overfitting;
- el modelo base debería estar equilibrado.

### 7.3 Qué documentar

Incluir una tabla tipo:

| Modelo | Filtros | Dropout | Train F1 | Val F1 |
|--------|---------|---------|----------|--------|
| Chico  | (16, 32)| 0.4     | ...      | ...    |
| Base   | (32, 64, 128)| 0.4 | ...   | ...    |
| Grande | (64, 128, 256)| 0.0 | ...  | ...    |

Y un gráfico del experimento de capacidad.

---

## 8. Prueba de fuga (opcional pero muy recomendada)

### 8.1 Qué es

Comparar dos particiones:

- split por grupos;
- split aleatorio sin control de grupos.

### 8.2 Qué buscar

Si la partición aleatoria da mucho mejor rendimiento que la partición por grupos, eso puede indicar que la red estaba aprendiendo artefactos o patrones repetidos de las mismas lesiones.

### 8.3 Cómo documentarlo

Se puede incluir como sección opcional del informe para reforzar la importancia de la validación rigurosa.

---

## 9. Evaluación final sobre holdout

### 9.1 Qué debe hacerse

- separar un test final por grupos con `GroupShuffleSplit`;
- usar este conjunto solo una vez al final;
- no reutilizarlo para ajustar hiperparámetros.

### 9.2 Qué documentar

Incluir un resumen final con:

- balanced accuracy
- F1 macro
- ROC-AUC
- recall mel

Esto es la prueba que demuestra la generalización real del modelo.

---

## 10. Generación del informe técnico

### 10.1 Qué debe tener el informe

El informe debe incluir:

- justificación de la arquitectura;
- descripción del dataset;
- protocolo de partición por grupos;
- hiperparámetros;
- tabla de métricas por fold;
- curvas de pérdida;
- experimento de capacidad;
- análisis de underfitting/overfitting;
- resultado final en test;
- conclusiones sobre si la CNN generaliza o memoriza.

### 10.2 Secciones recomendadas

1. Descripción del problema y del dataset
2. Diseño de la CNN
3. Preprocesamiento y partición por grupos
4. Entrenamiento y validación cruzada
5. Resultados por fold
6. Experimento de capacidad
7. Evaluación final sobre test
8. Diagnóstico de sesgo y varianza
9. Conclusiones finales

---

## 11. Checklist final antes de entregar

### Modelo
- [ ] arquitectura CNN con `Conv2D`
- [ ] `padding="same"`
- [ ] uso de `stride` y pooling
- [ ] `MaxPooling2D` en cada bloque
- [ ] ReLU y softmax en salida

### Datos
- [ ] dataset cargado correctamente
- [ ] imágenes normalizadas
- [ ] `lesion_id` disponible
- [ ] split por grupos implementado

### Validación
- [ ] holdout final separado por grupos
- [ ] 5 folds con `StratifiedGroupKFold`
- [ ] asserts de no solapamiento
- [ ] class weights calculados solo sobre train

### Evaluación
- [ ] balanced accuracy
- [ ] F1 macro
- [ ] ROC-AUC
- [ ] recall para clase crítica
- [ ] matriz de confusión
- [ ] curvas de aprendizaje
- [ ] experimento de capacidad

### Informe
- [ ] explicación técnica de la arquitectura
- [ ] análisis de sesgo/varianza
- [ ] comparación entre modelos
- [ ] resultado final en test
- [ ] conclusiones basadas en evidencia

---

## 12. Resumen ejecutivo

Para que la CNN cumpla la consigna, no basta con que “use convoluciones”. Debe quedar claro que:

- la red está diseñada para extraer patrones locales y jerárquicos;
- la validación está protegida contra fuga de información;
- los resultados se evalúan con métricas apropiadas;
- el análisis empírico demuestra si el modelo está sesgado, sobreajustado o bien equilibrado.

Si todo esto queda implementado y documentado, entonces sí se cumple la consigna de manera sólida.
