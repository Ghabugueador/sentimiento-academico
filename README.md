# 4. Metodología y desarrollo

Este proyecto clasifica el sentimiento expresado en **evaluaciones escritas de artículos científicos en español**. La entrada es el comentario de una evaluación; la salida es `negativo`, `neutral` o `positivo`. Se comparan Logistic Regression, Multinomial Naive Bayes y Linear SVM mediante TF-IDF, validación cruzada y métricas de prueba. El desarrollo comprende las subsecciones **4.1 a 4.5**.

## 4.1 Dataset y etiquetado

Se utiliza **Paper Reviews**, publicado por Brian Keith en 2017 en el UCI Machine Learning Repository. El archivo original `reviews.json` contiene 405 evaluaciones de artículos de una conferencia de informática, escritas principalmente en español. La fuente distingue `orientation`, que expresa el sentimiento percibido por los autores del estudio al leer el comentario, de `evaluation`, que registra la valoración asignada por el revisor. [Ficha y atribución del dataset](https://archive.ics.uci.edu/dataset/410/paper+reviews), [DOI: 10.24432/C50G60](https://doi.org/10.24432/C50G60).

El script `scripts/descargar_dataset.py` descarga el archivo original, aplana su estructura JSON y conserva las **388 evaluaciones cuyo idioma es español**, sin traducir textos. El CSV `data/articulos_es.csv` mantiene el texto y la información necesaria para auditar la preparación. El notebook retira **6 textos vacíos** y utiliza **382 evaluaciones de 168 artículos**. Las evaluaciones en inglés quedan fuera del experimento.

| Campo | Uso |
| --- | --- |
| `texto` | Comentario principal de la evaluación; única entrada textual del clasificador. |
| `orientacion` | Anotación original utilizada para construir la etiqueta de sentimiento. |
| `articulo_id` | Identificador original del artículo; agrupa sus evaluaciones durante las particiones. |
| `evaluacion_id` e `idioma` | Identificación y comprobación de procedencia; no son características predictoras. |
| `fecha` y `anio` | Información temporal de auditoría; no son características predictoras. |

Se utiliza únicamente el comentario principal `text` del JSON. **No se concatena `remarks`**, que contiene comentarios adicionales destinados al comité. No se introducen la valoración del revisor, la decisión de aceptación, la confianza ni los identificadores como características predictoras.

La etiqueta `sentimiento` agrupa la orientación original en tres clases:

| `orientation` original | Etiqueta |
| --- | --- |
| −2 o −1 | `negativo` |
| 0 | `neutral` |
| +1 o +2 | `positivo` |

Después de retirar los textos vacíos, quedan **172 ejemplos negativos, 104 neutrales y 106 positivos**. El análisis supervisado aprende estas etiquetas a partir del comentario. La etiqueta neutral corresponde a la orientación 0 del corpus; no se obtiene mediante una regla nueva sobre las palabras.

La orientación anotada representa una interpretación humana del texto. Puede existir ambigüedad o desacuerdo, especialmente en evaluaciones que mezclan elogios y críticas. El modelo estima esa orientación; **no decide si un artículo debe aceptarse ni mide su calidad científica**.

### Procedencia y reproducción

`data/source.json` registra las direcciones de origen, la transformación, las huellas SHA-256 y los conteos de preparación. El archivo original `data/reviews.json` permite revisar la transformación, y `data/README.md` describe el esquema. La atribución y los cambios realizados se documentan en [data/ATTRIBUTION.md](data/ATTRIBUTION.md).

El dataset se distribuye bajo **[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)**. Los archivos original y preparado se incluyen con atribución a Brian Keith y al repositorio UCI. La adaptación realizada consiste en aplanar el JSON, seleccionar español y construir las tres etiquetas; no modifica los comentarios originales. Los datos españoles corresponden a **2010, 2013, 2014 y 2015**. Las anotaciones y textos pertenecen a un conjunto pequeño de evaluaciones de una conferencia, por lo que no representan todas las disciplinas ni todos los procesos de revisión científica.

## 4.2 Limpieza y vectorización

La limpieza aplica normalización Unicode NFC, decodifica entidades HTML, convierte el texto a minúsculas, retira URLs y etiquetas HTML y compacta espacios. Conserva letras Unicode, incluidas **tildes, ü y ñ**, y elimina números y puntuación. No aplica stemming ni lematización.

Se conservan negaciones como `no`, `nunca`, `ni` y `sin`; `stop_words=None` evita retirarlas mediante una lista de palabras vacías. Por ejemplo, “el método está justificado” y “el método no está justificado” pueden expresar valoraciones diferentes. Los ejemplos utilizados para explicar la limpieza y demostrar la predicción son redactados para el proyecto.

Antes de dividir los datos, se comprueban textos vacíos y duplicados. También se buscan textos idénticos después de normalizar, incluidas coincidencias con etiquetas distintas. La regla conserva la primera aparición de cada texto normalizado y registra las exclusiones. **En el dataset preparado no se encuentran duplicados exactos ni normalizados**, por lo que la limpieza mantiene las 382 evaluaciones. Esta comprobación evita compartir el mismo texto entre conjuntos, pero no detecta todas las paráfrasis.

La representación numérica utiliza `TfidfVectorizer`. TF-IDF pondera los términos según su frecuencia en cada documento y su frecuencia en los documentos del entrenamiento correspondiente. La matriz es dispersa. [Documentación oficial de TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

| Parámetro | Valor y propósito |
| --- | --- |
| `max_features` | `5000`: limita el vocabulario a 5.000 características. |
| `ngram_range` | `(1, 2)`: representa palabras y pares consecutivos de palabras. |
| `min_df` | `2`: requiere presencia en al menos dos documentos del entrenamiento correspondiente. |
| `max_df` | `0.95`: excluye términos presentes en más del 95 % de esos documentos. |
| `sublinear_tf` | `True`: reduce el efecto de repeticiones mediante una transformación logarítmica. |
| `stop_words` | `None`: conserva palabras funcionales y negaciones. |

Las celdas 12 a 14 explican TF-IDF con una transformación sobre el entrenamiento y muestran sus dimensiones. **La búsqueda de hiperparámetros recibe textos y aprende su propio TF-IDF dentro de cada `Pipeline`**; no recibe una matriz ajustada previamente con todos los documentos. El vocabulario y los pesos se aprenden solo con el entrenamiento de cada pliegue. [Prevención de fuga de información en scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Retirar números y puntuación puede perder información sobre resultados, referencias o énfasis. Estos modelos tampoco comprenden de manera completa la ironía, el contenido técnico ni el contexto de un artículo. Una evaluación puede contener aspectos favorables y desfavorables aunque reciba una única etiqueta global.

## 4.3 Partición

Las evaluaciones de un mismo artículo pueden compartir tema y vocabulario. Por ello, la división **agrupa por `articulo_id`**: todas las evaluaciones de un artículo quedan en un único conjunto. El identificador se utiliza para organizar la división y no entra al clasificador.

Se obtiene una partición de prueba con el primer pliegue de:

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

Esto produce una división aproximadamente 80/20 que intenta conservar las proporciones de las clases y respeta los grupos. Las proporciones no tienen que coincidir exactamente, porque los artículos son grupos indivisibles. [Documentación oficial de StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

La partición contiene **307 evaluaciones de 132 artículos para entrenamiento** y **75 evaluaciones de 36 artículos para prueba**. En prueba hay **33 negativos, 21 neutrales y 21 positivos**. Se comprueba que ningún artículo ni texto normalizado aparece en ambos conjuntos.

El conjunto de prueba queda reservado hasta terminar el ajuste y la selección del modelo. Dentro del entrenamiento se utiliza otra validación cruzada de cinco pliegues con `StratifiedGroupKFold`, semilla 42 y agrupación por artículo. La búsqueda recibe `groups` para respetar esta separación. Se comprueban las clases y la ausencia de artículos compartidos en cada pliegue.

La evaluación describe el desempeño sobre comentarios de **artículos no presentes en el entrenamiento** dentro de este corpus. No se realiza una partición temporal ni una separación por identidad de revisor. El identificador de evaluación es local a cada artículo y no identifica a una persona. La semilla permite reproducir la división con los mismos datos, orden y versiones de dependencias.

## 4.4 Modelos e hiperparámetros

Cada candidato utiliza un `Pipeline` formado por TF-IDF y un clasificador supervisado:

- **Regresión logística:** aprende pesos para las características del texto y calcula probabilidades de pertenencia a las clases. Se utiliza para clasificación aunque su nombre incluya “regresión”. [Fundamento y documentación](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
- **Naive Bayes multinomial:** aplica el teorema de Bayes y supone independencia condicional entre las características dada la clase. Puede trabajar con valores TF-IDF no negativos; el suavizado evita asignar probabilidad cero a términos no observados en una clase. [Fundamento y documentación](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes).
- **SVM lineal:** aprende fronteras lineales con regularización y un criterio de separación por margen. `LinearSVC` trata las tres clases mediante clasificadores de una clase frente al resto. [Fundamento y documentación](https://scikit-learn.org/stable/modules/svm.html#classification).

| Modelo | Configuración fija | Valores evaluados |
| --- | --- | --- |
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` | `C`: 0.1, 1 y 10 |
| Multinomial Naive Bayes | Configuración restante predeterminada | `alpha`: 0.1, 0.5 y 1 |
| Linear SVM (`LinearSVC`) | `class_weight="balanced"`, `max_iter=10000`, `dual="auto"` | `C`: 0.1, 1 y 10 |

`C` controla inversamente la intensidad de regularización: un valor menor impone mayor regularización. `alpha` controla el suavizado de Naive Bayes. Los pesos equilibrados modifican el peso relativo de las clases durante el ajuste; no generan comentarios artificiales ni equilibran los conjuntos de evaluación.

`GridSearchCV` compara tres valores por modelo en cinco pliegues agrupados, con `scoring="f1_macro"`. Se realizan 45 ajustes de validación, además del reajuste del mejor candidato de cada familia sobre todo el entrenamiento. El modelo final se selecciona por el **mayor F1 macro medio de validación cruzada antes de consultar la prueba**. [Documentación oficial de GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

Se añade un `DummyClassifier(strategy="most_frequent")` como referencia: predice siempre la clase más frecuente del entrenamiento. Su resultado permite comprobar si los modelos aportan mejora frente a una decisión que no usa el contenido del texto.

En la prueba se calculan estas métricas:

| Métrica | Interpretación |
| --- | --- |
| Accuracy | Proporción de predicciones correctas sobre el total. |
| Precision macro | Media de la precisión de cada clase, con el mismo peso para las tres. |
| Recall macro | Media de la proporción de ejemplos recuperados de cada clase. |
| F1 macro | Media del F1 por clase, que combina Precision y Recall. |

También se generan el informe por clase y matrices de confusión con conteos y proporciones por clase real. La tabla de prueba describe los candidatos ya ajustados; no se utiliza para volver a elegir hiperparámetros ni cambiar el criterio de selección.

### Resultados y límites de interpretación

La ejecución selecciona **Linear SVM con `C=10`**, cuyo F1 macro medio de validación cruzada es **0,5133**. Logistic Regression obtiene **0,5123** en esa validación, una diferencia pequeña. En el conjunto de prueba se observan estos valores:

| Modelo | Accuracy | F1 macro de prueba | Seleccionado por CV |
| --- | --- | --- | --- |
| Logistic Regression | 58,67 % | 0,5119 | No |
| Multinomial Naive Bayes | 50,67 % | 0,3766 | No |
| Linear SVM | 46,67 % | 0,3877 | Sí |
| Referencia: clase mayoritaria | 44,00 % | 0,2037 | No |

Estas cifras proceden de [resultados/metricas.json](resultados/metricas.json) y [resultados/metricas_prueba.csv](resultados/metricas_prueba.csv). Logistic Regression presenta mejores métricas de prueba en esta partición, pero la selección realizada antes de consultar la prueba se mantiene: cambiarla después utilizaría el conjunto reservado como parte de la decisión.

Para el modelo seleccionado, el F1 por clase es **0,6410 en negativo, 0,0606 en neutral y 0,4615 en positivo**. El menor F1 corresponde a neutral; el modelo recupera solo 1 de los 21 comentarios neutrales de prueba. Los valores completos están en [resultados/reporte_clasificacion.csv](resultados/reporte_clasificacion.csv). Los resultados muestran una limitación real del clasificador y no justifican presentar la demo como un sistema fiable para tomar decisiones editoriales.

El conjunto de prueba tiene solo 75 evaluaciones. Unos pocos errores pueden cambiar apreciablemente las métricas. No se calculan intervalos de confianza ni pruebas de significación, por lo que una diferencia pequeña entre candidatos no demuestra superioridad general. Tampoco se valida el uso para otras conferencias, otros idiomas o disciplinas distintas. Los ejemplos de la demo no forman parte del conjunto de prueba y no constituyen una garantía de acierto.

## 4.5 Explicación del código

El desarrollo sigue un recorrido: cargar y comprobar los datos, crear las etiquetas, limpiar el texto, separar los artículos, aprender la representación numérica, ajustar los modelos, evaluar y clasificar un comentario nuevo. **Cada una de las 23 celdas de código del notebook [notebooks/sentiment_analysis.ipynb](notebooks/sentiment_analysis.ipynb) tiene una explicación inmediatamente encima**, con su objetivo, operaciones y lectura de las salidas. La numeración cuenta celdas de código, no bloques de texto ni el número de ejecuciones que Jupyter muestra entre corchetes.

Para localizar rápidamente la parte de los modelos, distinga estos momentos. **El proyecto entrena sus modelos con el dataset; no carga un modelo previamente entrenado desde un archivo.**

| Momento | Celda | Operación principal | Estado del modelo |
| --- | --- | --- | --- |
| Importar las clases | 1 | `from sklearn... import ...` | Las herramientas quedan disponibles para utilizarlas. |
| Crear los candidatos | 15 | `LogisticRegression(...)`, `MultinomialNB()` y `LinearSVC(...)` | Son objetos configurados, todavía sin entrenar. |
| Entrenar y comparar configuraciones | 17 | `busqueda.fit(X_train_text, y_train, groups=grupos_train)` | Cada candidato aprende dentro de los pliegues de entrenamiento. |
| Recuperar el candidato ajustado de cada familia | 17 | `busqueda.best_estimator_` | Con `refit=True`, el Pipeline ganador se reajusta con todo el entrenamiento. |
| Seleccionar la familia final | 17 | `mejor_modelo = mejores_modelos[mejor_nombre]` | Se conserva el Pipeline con mayor F1 macro de validación cruzada. |
| Predecir comentarios | 18 y 23 | `modelo.predict(...)` y `mejor_modelo.predict(...)` | Se aplica lo aprendido, sin volver a entrenar. |

### Guía para localizar cada paso

| Celda | Qué hace | Qué revisar |
| --- | --- | --- |
| 1 | Importa herramientas y configura rutas, semilla y gráficos. | Versiones y carpeta del proyecto. |
| 2 | Verifica el origen y carga el CSV español. | 388 evaluaciones, incluidas las vacías. |
| 3 | Comprueba columnas, tipos e identificadores. | Una fila por evaluación; varios comentarios por artículo. |
| 4 | Conserva texto, orientación e identificadores. | Solo el texto entra al clasificador. |
| 5 | Excluye textos vacíos y valida la orientación. | Seis exclusiones; quedan 382 evaluaciones. |
| 6 | Revisa duplicados exactos de texto y orientación. | Conteo de repeticiones. |
| 7 | Convierte orientación en negativo, neutral o positivo. | Tabla de correspondencia de las etiquetas. |
| 8 | Cuenta y dibuja la distribución de clases. | 172 negativos, 104 neutrales y 106 positivos. |
| 9 | Limpia español y revisa textos equivalentes. | Negaciones conservadas; exclusiones y conflictos registrados. |
| 10 | Define `X`, `y` y `grupos`. | Texto, respuesta esperada y artículo alineados. |
| 11 | Separa entrenamiento y prueba por artículo. | 307 y 75 evaluaciones; artículos sin solapamiento. |
| 12 | Configura palabras, bigramas y filtros TF-IDF. | Qué significa cada opción del vectorizador. |
| 13 | Demuestra `fit_transform` y `transform`. | Aprende solo con entrenamiento. |
| 14 | Inspecciona la representación numérica. | Filas, columnas, valores no negativos y vocabulario. |
| 15 | Crea regresión logística, Naive Bayes y SVM. | Cómo aprende cada algoritmo y sus opciones fijas. |
| 16 | Define hiperparámetros y cinco pliegues internos. | Validación agrupada dentro de entrenamiento. |
| 17 | Busca configuraciones y selecciona mediante CV. | SVM, `C=10`, F1 macro CV 0,5133. |
| 18 | Calcula métricas de prueba y la referencia mayoritaria. | Accuracy, Precision macro, Recall macro y F1 macro. |
| 19 | Muestra la comparación en tabla y gráfico. | Métricas de prueba sin cambiar la selección previa. |
| 20 | Guarda el registro del experimento. | `metricas.json`; no contiene el modelo entrenado. |
| 21 | Examina Precision, Recall, F1 y soporte por clase. | Neutral: F1 0,0606 y 1 de 21 recuperados. |
| 22 | Muestra conteos y proporciones de los errores. | Filas reales, columnas predichas y diagonal de aciertos. |
| 23 | Predice el sentimiento de `mi_evaluacion`. | Demostración de la práctica con el modelo ya entrenado. |

### Variables y operaciones que conviene reconocer

| Nombre u operación | Significado |
| --- | --- |
| `df_original` / `df` | Tabla cargada / tabla de trabajo que se prepara para modelar. |
| `auditoria` | Diccionario con cantidades y exclusiones del proceso. |
| `X` / `y` / `grupos` | Comentarios limpios / etiquetas conocidas / identificadores de artículo para organizar las particiones. |
| `X_train_text` / `X_test_text` | Comentarios limpios de entrenamiento / prueba. |
| `y_train` / `y_test` | Etiquetas conocidas de entrenamiento / prueba. |
| `grupos_train` / `grupos_test` | Artículos de entrenamiento / prueba; no son entradas predictoras. |
| `X_train` / `X_test` | Matrices TF-IDF didácticas; no son la entrada de GridSearchCV. |
| `modelos` | Diccionario con los tres clasificadores configurados antes de entrenar. |
| `pipeline` | Secuencia que conecta el vectorizador `tfidf` con el clasificador `modelo`. |
| `busqueda` | Objeto GridSearchCV que compara hiperparámetros mediante validación cruzada. |
| `puntajes_cv` | Mejor F1 macro medio de validación cruzada de cada familia. |
| `mejores_modelos` / `mejor_modelo` | Candidatos reajustados / Pipeline seleccionado mediante CV. |
| `y_pred_final` | Predicciones del modelo seleccionado sobre la prueba. |
| `def` / `return` | Define una función / entrega su resultado al código que la utiliza. |
| `for` | Repite operaciones para cada elemento de una colección. |
| `assert` | Detiene la ejecución cuando no se cumple una comprobación. |
| `fit` | Aprende usando los datos proporcionados. |
| `transform` | Aplica una representación ya aprendida sin volver a ajustarla. |
| `fit_transform` | Aprende la representación y transforma los mismos datos. |
| `predict` | Utiliza un modelo entrenado para obtener etiquetas. |
| `display` / `print` / `to_csv` | Muestra tablas o texto enriquecido / imprime mensajes / guarda tablas. |

### Partes importantes del código para la exposición

Los fragmentos siguientes seleccionan las líneas relevantes del notebook. Sirven para explicar el desarrollo junto con las celdas completas; deben ejecutarse dentro de su secuencia, porque utilizan variables creadas anteriormente.

#### 1. Preparación y carga de los datos — celdas 1 a 6

La celda 1 importa bibliotecas y define `SEMILLA = 42`, `N_JOBS = 2` y `CLASES`. `SEMILLA` permite repetir las particiones con los mismos datos, orden y versiones. `N_JOBS` limita a dos los trabajos paralelos de la búsqueda. `CLASES` fija el orden negativo, neutral y positivo en los informes y matrices. `RAIZ` localiza la carpeta del proyecto mediante `Path`, y `SALIDA` identifica `resultados/`.

En la celda 2 se leen los datos:

```python
ruta_dataset = RAIZ / "data" / "articulos_es.csv"
df_original = pd.read_csv(ruta_dataset, keep_default_na=False)
```

`ruta_dataset` identifica el CSV y `pd.read_csv` lo convierte en una tabla de pandas. `keep_default_na=False` conserva los campos vacíos como cadenas, para identificarlos y excluirlos de forma explícita. La tabla inicial contiene 388 evaluaciones en español.

Antes de leer el CSV, la misma celda ejecuta `scripts/descargar_dataset.py` mediante `subprocess.run` y el Python del kernel, `sys.executable`. El script verifica el original y prepara el CSV. La celda consulta `source.json`, calcula SHA-256 y compara la huella del CSV con la registrada. Esta comprobación detecta cambios en los bytes del archivo; no evalúa la calidad de las etiquetas.

Las celdas 3 a 6 comprueban el esquema, conservan las columnas necesarias y retiran textos vacíos y duplicados exactos. `df_original` conserva la tabla cargada y `df` contiene la tabla de trabajo. `auditoria` registra los conteos. Los `assert` detienen el desarrollo si una condición, como la unicidad de los identificadores, no se cumple.

**Para exponer:** “Primero verificamos la procedencia y cargamos el CSV. Cada fila representa una evaluación. Después excluimos seis comentarios vacíos y conservamos 382 para el análisis”.

#### 2. Etiquetas, limpieza y variables de entrada — celdas 7, 9 y 10

La celda 7 define `etiquetar_sentimiento()` y la aplica a la orientación original:

```python
df["sentimiento"] = df["orientacion"].apply(etiquetar_sentimiento)
```

`apply` llama a la función para cada valor de `orientacion`. La función asigna negativo a −2 y −1, neutral a 0 y positivo a 1 y 2. Estas etiquetas son las respuestas conocidas que permiten el aprendizaje supervisado. Esta asignación no es una predicción del modelo ni una búsqueda de palabras positivas o negativas.

La celda 9 define la limpieza utilizada tanto para preparar el dataset como para procesar un comentario nuevo:

```python
def limpiar_texto(texto):
    texto = unicodedata.normalize("NFC", html.unescape(str(texto))).lower()
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = "".join(c if c.isalpha() or c.isspace() else " " for c in texto)
    return re.sub(r"\s+", " ", texto).strip()
```

La primera línea decodifica entidades HTML, unifica la representación Unicode y convierte a minúsculas. Los dos `re.sub` siguientes retiran etiquetas HTML y URLs. `isalpha()` conserva letras Unicode, incluidas tildes y ñ; `isspace()` conserva espacios. Otros caracteres se sustituyen por espacios. La última línea compacta espacios repetidos y retira los de los extremos. La función no elimina palabras por considerarlas frecuentes, por lo que conserva negaciones como “no”.

La función se aplica con `df["texto"].apply(limpiar_texto)` para crear `texto_limpio`. Después se revisan textos que quedaron vacíos y repeticiones normalizadas. La celda 10 separa las funciones de cada columna:

```python
X = df["texto_limpio"]
y = df["sentimiento"]
grupos = df["articulo_id"]
```

`X` contiene la información que recibirá el clasificador. `y` contiene lo que debe aprender a predecir. `grupos` organiza las particiones por artículo; no se concatena con el texto ni se introduce como característica. El orden de las tres series debe mantenerse alineado para que cada comentario corresponda a su etiqueta y artículo.

**Para exponer:** “La orientación original proporciona la respuesta conocida. Limpiamos el comentario sin eliminar las negaciones. El modelo recibe solo texto; los artículos sirven para organizar la separación”.

#### 3. Reserva de prueba por artículo — celda 11

```python
separador = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
indices_train, indices_test = next(separador.split(X, y, groups=grupos))
X_train_text, X_test_text = X.iloc[indices_train], X.iloc[indices_test]
y_train, y_test = y.iloc[indices_train], y.iloc[indices_test]
grupos_train, grupos_test = grupos.iloc[indices_train], grupos.iloc[indices_test]
assert set(grupos_train).isdisjoint(grupos_test)
assert set(X_train_text).isdisjoint(X_test_text)
```

`split` genera índices de particiones que respetan los grupos y tratan de conservar las proporciones de las clases. `next` toma la primera división de los cinco pliegues como reserva de prueba. `iloc` selecciona las filas por posición y aplica los mismos índices a textos, etiquetas y artículos.

Quedan 307 evaluaciones para entrenamiento y 75 para prueba. `isdisjoint` comprueba que no haya artículos ni textos iguales compartidos entre ambos conjuntos. La prueba permanece reservada mientras se buscan hiperparámetros; no se utiliza para decidir el vocabulario ni seleccionar el modelo.

**Para exponer:** “Separamos artículos completos. Así comprobamos el resultado sobre evaluaciones de artículos diferentes de los utilizados para entrenar”.

#### 4. Conversión del texto a números — celdas 12 a 14

Los clasificadores utilizan características numéricas. La celda 12 configura el vectorizador:

```python
config_tfidf = dict(max_features=5000, ngram_range=(1, 2), min_df=2,
                    max_df=0.95, stop_words=None, sublinear_tf=True)
tfidf = TfidfVectorizer(**config_tfidf)
```

`config_tfidf` reúne los parámetros y `**config_tfidf` los entrega como argumentos al constructor. Crear `tfidf` todavía no aprende un vocabulario. Las opciones representan palabras y pares de palabras, limitan el vocabulario y conservan negaciones. Los valores y su propósito están detallados en 4.2.

La celda 13 muestra la diferencia entre aprender y aplicar la transformación:

```python
tfidf_demostracion = clone(tfidf)
X_train = tfidf_demostracion.fit_transform(X_train_text)
X_test = tfidf_demostracion.transform(X_test_text)
```

`clone` crea un vectorizador independiente con la misma configuración. `fit_transform` aprende el vocabulario y los pesos IDF con el entrenamiento y obtiene su matriz. `transform` representa la prueba usando ese vocabulario, sin aprender uno nuevo. Cada fila corresponde a un comentario y cada columna a una característica textual. La celda 14 muestra las dimensiones y algunos términos del vocabulario.

**Estas matrices son didácticas.** La búsqueda de la celda 17 recibe `X_train_text`, no `X_train`. Sus Pipelines ajustan vectorizadores independientes dentro de cada pliegue. Utilizar un vocabulario aprendido con todas las filas antes de validar revelaría información de la validación.

**Para exponer:** “TF-IDF transforma palabras y pares de palabras en valores numéricos. Mostramos cómo funciona y, durante la búsqueda, lo aprendemos de nuevo dentro de cada pliegue de entrenamiento”.

#### 5. Importación y creación de los tres modelos — celdas 1 y 15

La celda 1 importa las clases de scikit-learn:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
```

Importar permite utilizar las implementaciones de los algoritmos. No descarga pesos ni restaura un modelo ya entrenado. La celda 15 crea los candidatos:

```python
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEMILLA),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight="balanced", max_iter=10000, dual="auto", random_state=SEMILLA),
}
```

`modelos` es un diccionario: cada clave identifica una familia y cada valor es un objeto que conserva su configuración. Regresión logística aprende pesos asociados a las características; Naive Bayes utiliza un modelo probabilístico; SVM aprende fronteras de decisión por margen. En este punto ninguno de los tres candidatos ha visto los comentarios.

`class_weight="balanced"` ajusta el peso de las clases durante el entrenamiento de regresión logística y SVM. `max_iter` establece un límite de iteraciones del optimizador; no representa el número de comentarios ni el número de pliegues. `dual="auto"` permite a LinearSVC elegir la formulación de optimización compatible con los datos y parámetros. Las opciones no garantizan por sí solas un mejor resultado.

**Para exponer:** “Aquí creamos tres candidatos con sus parámetros iniciales. Todavía no están entrenados; aprenderán al ejecutar `fit` en la búsqueda de la celda 17”.

#### 6. Entrenamiento, búsqueda y selección final — celdas 16 y 17

La celda 16 define las configuraciones que se compararán:

```python
parametros = {
    "Logistic Regression": {"modelo__C": [0.1, 1, 10]},
    "Multinomial Naive Bayes": {"modelo__alpha": [0.1, 0.5, 1.0]},
    "Linear SVM": {"modelo__C": [0.1, 1, 10]},
}
```

Los nombres contienen dos guiones bajos: `modelo__C` significa “cambiar el parámetro `C` del paso llamado `modelo`”. `C` controla inversamente la regularización en regresión logística y SVM. `alpha` controla el suavizado en Naive Bayes. Esta celda también crea `cv`, con cinco pliegues agrupados dentro del entrenamiento, y comprueba que no compartan artículos.

El siguiente fragmento de la celda 17 reúne las operaciones centrales del bucle. Se omiten los registros de tiempos y archivos CSV:

```python
for nombre, modelo in modelos.items():
    pipeline = Pipeline([("tfidf", clone(tfidf)), ("modelo", clone(modelo))])
    busqueda = GridSearchCV(
        pipeline, parametros[nombre], scoring="f1_macro", cv=cv,
        n_jobs=N_JOBS, pre_dispatch=N_JOBS, refit=True, error_score="raise",
    )
    with parallel_backend("threading"), threadpool_limits(limits=1):
        busqueda.fit(X_train_text, y_train, groups=grupos_train)
    mejores_modelos[nombre] = busqueda.best_estimator_
    puntajes_cv[nombre] = float(busqueda.best_score_)
```

`for` recorre las tres familias. `Pipeline` conecta dos pasos: `tfidf` representa el texto y `modelo` aprende a clasificar esa representación. Las copias creadas con `clone` evitan reutilizar un objeto previamente ajustado. La limpieza ya se realizó antes; no forma parte de este Pipeline.

**El entrenamiento real comienza en `busqueda.fit(...)`.** `X_train_text` aporta comentarios, `y_train` aporta respuestas conocidas y `groups=grupos_train` organiza los pliegues. Para cada configuración, GridSearchCV aprende TF-IDF y ajusta el clasificador solo con la parte de entrenamiento del pliegue; calcula F1 macro sobre su validación. La parte reservada en `X_test_text` no interviene. [Funcionamiento de Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

`scoring="f1_macro"` establece qué métrica se busca mejorar. `refit=True` reajusta el candidato ganador de cada familia con los 307 comentarios de entrenamiento. `n_jobs` y `pre_dispatch` limitan los trabajos paralelos; el bloque `with` controla el backend y los hilos internos. `error_score="raise"` hace visible un error de ajuste en vez de sustituirlo por una puntuación. Se realizan 45 ajustes de validación —tres familias, tres valores y cinco pliegues— y tres reajustes finales.

`best_estimator_` devuelve el Pipeline ganador de esa familia, ya ajustado. `best_score_` devuelve su F1 macro medio de validación cruzada; no contiene el resultado de prueba. Se almacenan en `mejores_modelos` y `puntajes_cv`, respectivamente. [Atributos y reajuste de GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

Al finalizar el bucle se escoge la familia final:

```python
mejor_nombre = max(puntajes_cv, key=puntajes_cv.get)
mejor_modelo = mejores_modelos[mejor_nombre]
```

`max` compara los valores de `puntajes_cv`, utilizando `get` para consultar el puntaje de cada nombre. `mejor_nombre` guarda el nombre ganador y `mejor_modelo` recupera su Pipeline entrenado. En la ejecución guardada se selecciona Linear SVM con `C=10` y F1 macro CV de 0,5133. La selección se completa antes de examinar la prueba.

**Para exponer:** “Comparamos configuraciones usando validación cruzada dentro del entrenamiento. Cada pliegue aprende su TF-IDF. Luego reajustamos el ganador de cada familia y elegimos el de mayor F1 macro de validación”.

#### 7. Predicción sobre prueba y cálculo de métricas — celdas 18, 19, 21 y 22

La celda 18 recorre `mejores_modelos` y utiliza estas líneas dentro del bucle:

```python
predicho = modelo.predict(X_test_text)
predicciones[nombre] = predicho
resultados.append({"Modelo": nombre, **calcular_metricas(y_test, predicho)})
```

`predict` aplica el TF-IDF ya aprendido y el clasificador ajustado a los 75 comentarios de prueba. `predicho` contiene las etiquetas estimadas. `predicciones` conserva las salidas de cada familia para examinarlas después. `calcular_metricas` compara las etiquetas conocidas `y_test` con esas predicciones. `**` incorpora el diccionario de métricas a la fila de resultados.

```python
def calcular_metricas(real, predicho):
    return {
        "Accuracy": accuracy_score(real, predicho),
        "Precision_macro": precision_score(real, predicho, labels=CLASES, average="macro", zero_division=0),
        "Recall_macro": recall_score(real, predicho, labels=CLASES, average="macro", zero_division=0),
        "F1_macro": f1_score(real, predicho, labels=CLASES, average="macro", zero_division=0),
    }
```

Accuracy mide aciertos sobre el total. Para una clase, Precision indica qué proporción de las predicciones de esa clase son correctas; Recall indica qué proporción de sus ejemplos reales se recuperan. F1 combina ambas. `average="macro"` calcula cada métrica por clase y promedia con el mismo peso para las tres. F1 macro promedia los F1 de las clases; no se obtiene combinando Precision macro con Recall macro. `zero_division=0` asigna cero cuando un cociente queda indefinido. [Definición de F1 y promedio macro](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

El `DummyClassifier` de la misma celda es una referencia que siempre elige la clase mayoritaria. Su `fit` aprende cuál es esa clase a partir de `y_train`; no vuelve a entrenar los tres modelos. La celda 19 muestra la comparación, manteniendo la selección realizada por validación cruzada.

La celda 21 recupera `y_pred_final = predicciones[mejor_nombre]`. `classification_report` separa Precision, Recall, F1 y soporte por clase; soporte es la cantidad de ejemplos reales de esa clase. La celda 22 construye `confusion_matrix(y_test, y_pred_final, labels=CLASES)`. Las filas indican la clase real, las columnas la predicha y la diagonal contiene los aciertos. La matriz normalizada divide cada fila por su soporte para mostrar proporciones.

Para comprobar la interpretación con la salida guardada, SVM acierta 35 de 75 comentarios: Accuracy es 46,67 %. Su F1 macro de prueba es 0,3877. Recupera solo 1 de los 21 neutrales. El informe y la matriz permiten explicar este comportamiento además de presentar una cifra global.

**Para exponer:** “Evaluamos los modelos sobre datos reservados. Accuracy muestra el acierto global; F1 macro y el informe por clase permiten observar el rendimiento de las tres clases por separado”.

#### 8. Registro de resultados y estado del modelo — celda 20

La celda 20 crea `resumen` con la fuente, huellas, versiones, semilla, partición, configuración y métricas, y lo escribe en JSON:

```python
(SALIDA / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
```

`json.dumps` convierte el diccionario en texto. `indent=2` facilita su lectura y `ensure_ascii=False` conserva los caracteres españoles. `write_text` guarda ese texto en `resultados/metricas.json` con codificación UTF-8. Otros bloques guardan tablas con `to_csv` y figuras con `savefig`.

**Estos archivos documentan el experimento; no almacenan el objeto `mejor_modelo`.** El notebook guarda código y salidas, pero el Pipeline entrenado permanece en la memoria del kernel durante la sesión. El proyecto no utiliza `joblib.dump`, `joblib.load` ni un archivo de pesos para restaurarlo. Reiniciar o cerrar el kernel elimina las variables: hay que ejecutar de nuevo las celdas en orden para reconstruir el modelo, aunque todavía se vean las salidas guardadas.

**Para exponer:** “Guardamos evidencias para revisar el experimento. El modelo entrenado sigue en la memoria de Jupyter y se reutiliza mientras esa sesión permanezca activa”.

#### 9. Demostración de la práctica con un comentario nuevo — celda 23

La celda 23 comprueba primero que existan `pd`, `limpiar_texto` y `mejor_modelo`. Si faltan, muestra un mensaje que indica ejecutar las celdas desde el inicio. Después define la función de predicción:

```python
def predecir_sentimiento(texto):
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("Escriba una evaluación no vacía en español.")
    texto_limpio = limpiar_texto(texto)
    vector = mejor_modelo.named_steps["tfidf"].transform([texto_limpio])
    if not texto_limpio or vector.nnz == 0:
        raise ValueError("El comentario no contiene términos conocidos por el modelo.")
    return str(mejor_modelo.predict([texto_limpio])[0])
```

`isinstance` comprueba que la entrada sea una cadena y `strip` permite detectar texto vacío. `limpiar_texto` aplica la misma preparación utilizada para los comentarios del dataset. `named_steps["tfidf"]` accede al vectorizador entrenado dentro del Pipeline. Su `transform` representa el comentario con el vocabulario existente. `vector.nnz` cuenta las posiciones con valor distinto de cero; si es cero, ninguna característica conocida representa esa entrada.

En `predict([texto_limpio])`, los corchetes forman una colección con un solo comentario, que es el formato esperado. El Pipeline aplica de nuevo su transformación y después el clasificador, sin ajustar sus parámetros. `[0]` extrae la primera y única etiqueta; `str` la entrega como texto. La transformación anterior se utiliza para comprobar el vocabulario, mientras que `predict` realiza la clasificación completa. Este control no detecta automáticamente el idioma ni garantiza un acierto.

Para la exposición, cambie únicamente el contenido de `mi_evaluacion` y ejecute esa celda con el kernel entrenado:

```python
mi_evaluacion = "El artículo no explica la metodología y sus conclusiones no están justificadas."
print("Sentimiento predicho:", predecir_sentimiento(mi_evaluacion))
```

Los ejemplos escritos en la demo no forman parte del conjunto de prueba. La predicción representa el sentimiento estimado del comentario; no determina la aceptación ni la calidad científica del artículo.

**Para exponer:** “Limpiamos el comentario nuevo con la misma función, comprobamos que tenga términos conocidos y utilizamos el Pipeline seleccionado. Esta celda predice sin volver a entrenar”.

### Preguntas para explicar el desarrollo

- **¿Se clasifica el artículo completo?** No. Se clasifica el sentimiento de un comentario escrito sobre el artículo.
- **¿Por qué usar orientación y no valoración?** La orientación anota cómo se percibe el texto al leerlo. La valoración del revisor puede diferir del sentimiento expresado.
- **¿Por qué excluir las otras columnas como entradas?** El objetivo es aprender desde el texto. Las decisiones, las puntuaciones y la orientación revelarían información ajena a esa entrada o directamente relacionada con la respuesta.
- **¿Por qué agrupar por artículo?** Para evaluar sobre artículos diferentes y evitar que las evaluaciones de un mismo artículo queden repartidas entre entrenamiento y evaluación.
- **¿Por qué mantener las negaciones?** Porque pueden invertir o matizar una valoración; los bigramas ayudan a representar combinaciones como “no aporta”.
- **¿Por qué utilizar Pipeline?** Para ajustar TF-IDF exclusivamente con el entrenamiento de cada pliegue durante la búsqueda.
- **¿Por qué F1 macro?** Porque asigna la misma importancia a las tres clases aunque sus frecuencias sean distintas.
- **¿Qué significa un valor fuera de la diagonal en la matriz de confusión?** Es una predicción que confunde la clase real con la clase indicada por la columna.
- **¿Una predicción positiva implica aceptar el artículo?** No. El clasificador no reemplaza la evaluación científica ni la decisión editorial.

### Archivos y ejecución

```text
sentimiento-academico/
├── data/
│   ├── reviews.json             # Archivo original UCI; CC BY 4.0
│   ├── articulos_es.csv         # Adaptación española con atribución
│   ├── source.json              # Procedencia y huellas de los datos
│   ├── ATTRIBUTION.md
│   └── README.md
├── notebooks/
│   ├── sentiment_analysis.ipynb
│   └── sentiment_analysis.html  # Vista de lectura generada al ejecutar
├── scripts/
│   ├── descargar_dataset.py
│   ├── ejecutar_notebook.py
│   └── comprobar_proyecto.py
├── resultados/                 # Métricas, tablas y figuras de la ejecución
├── abrir_notebook.bat          # Abrir JupyterLab en Windows
├── abrir_notebook.sh           # Abrir JupyterLab en Linux
├── requirements.txt
├── requirements-lock.txt
├── .gitattributes              # Conserva los bytes de datos y las líneas del .sh
├── .gitignore
└── README.md
```

El mismo notebook y los mismos scripts se utilizan en **Windows y Linux**. Cada compañero debe crear su propio entorno con **Python 3.12**; no copie la carpeta `.venv` de otro equipo o sistema operativo. Abra una terminal en la carpeta `sentimiento-academico`, donde se encuentran este README y `requirements-lock.txt`.

**Preparación en Windows (PowerShell).** Con Python 3.12 instalado y disponible mediante el lanzador `py`, ejecute:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\abrir_notebook.bat
```

Después de preparar el entorno una vez, puede abrir `abrir_notebook.bat` con doble clic. Si `py -3.12` no encuentra Python, utilice la ruta al ejecutable de Python 3.12; por ejemplo, `& "C:\ruta\a\python.exe" -m venv .venv` en PowerShell.

**Preparación en Linux (terminal).** Compruebe que está instalado Python 3.12 con soporte para crear entornos virtuales y ejecute:

```sh
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
sh abrir_notebook.sh
```

Si su comando `python3` corresponde a Python 3.12, puede utilizar `python3 -m venv .venv`. Si falta Python 3.12 o su módulo `venv`, instálelos con el gestor de paquetes de su distribución antes de continuar. En Ubuntu o Debian, el nombre y la disponibilidad del paquete dependen de la versión del sistema y de cómo se instaló Python; compruebe que el soporte `venv` corresponda al Python 3.12 utilizado.

Después de preparar el entorno una vez, utilice `sh abrir_notebook.sh` para abrir la práctica. No necesita activar `.venv` ni cambiar los permisos del archivo: el lanzador utiliza directamente el Python del entorno del proyecto. Los lanzadores comprueban que el entorno exista; la instalación de dependencias se realiza con los comandos anteriores.

`requirements-lock.txt` fija las versiones e incluye marcadores del sistema operativo: instala `pywinpty` en Windows y `pexpect` y `ptyprocess` en Linux. Las dependencias compartidas, como `tzdata`, se instalan en ambos sistemas. `.gitattributes` mantiene los bytes de `data/reviews.json`, `data/articulos_es.csv` y `data/source.json` al compartirlos mediante Git, para conservar los datos y sus verificaciones de integridad; también mantiene finales de línea LF en el lanzador de Linux. No cambie los archivos originales ni las huellas SHA-256 para corregir un error de verificación.

Si prefiere abrir JupyterLab sin el lanzador, utilice el comando de su sistema desde la carpeta del proyecto:

| Sistema | Comando |
| --- | --- |
| Windows / PowerShell | `.\.venv\Scripts\python.exe -m jupyterlab notebooks/sentiment_analysis.ipynb` |
| Linux | `.venv/bin/python -m jupyterlab notebooks/sentiment_analysis.ipynb` |

En JupyterLab, ejecute **Restart Kernel and Run All Cells** para cargar los datos y entrenar los modelos en orden. Espere hasta que el kernel vuelva a estar inactivo y termine la última celda. Ejecutar únicamente la demo en un kernel nuevo produce errores porque todavía no existen las funciones ni el modelo entrenado.

Para probar otro comentario, vaya a la **celda 23** y cambie solo el texto entre comillas de `mi_evaluacion`. Por ejemplo:

```python
mi_evaluacion = "La metodología está bien justificada y los resultados aportan evidencia clara."
```

Pulse **Shift + Enter** con esa celda seleccionada. La función limpia el texto, comprueba que contiene términos del vocabulario y reutiliza el `Pipeline` entrenado. No vuelve a entrenar los algoritmos. Se rechazan textos vacíos o sin términos conocidos, pero este control no detecta automáticamente el idioma.

Para ejecutar el desarrollo completo y guardar el notebook con salidas y su vista HTML, utilice el comando de su sistema:

Windows / PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/ejecutar_notebook.py
```

Linux:

```sh
.venv/bin/python scripts/ejecutar_notebook.py
```

Los archivos de datos se incluyen con atribución. La celda 2 ejecuta el descargador para verificar el JSON y reconstruir el CSV y sus metadatos, aunque el CSV ya exista. Con `data/reviews.json` incluido y su huella correcta, esta preparación es local; si falta el original, requiere conexión a Internet. También puede ejecutarla directamente:

Windows / PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/descargar_dataset.py
```

Linux:

```sh
.venv/bin/python scripts/descargar_dataset.py
```

Para comprobar la integridad de los datos y las evidencias del proyecto, sin entrenar ni modificar archivos:

Windows / PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/comprobar_proyecto.py
```

Linux:

```sh
.venv/bin/python scripts/comprobar_proyecto.py
```
