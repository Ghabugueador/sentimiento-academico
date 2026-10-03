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
| `X_train_text` / `X_test_text` | Comentarios limpios de entrenamiento / prueba. |
| `y_train` / `y_test` | Etiquetas conocidas de entrenamiento / prueba. |
| `X_train` / `X_test` | Matrices TF-IDF didácticas; no son la entrada de GridSearchCV. |
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

### Cómo explicar la práctica

Las etiquetas se conocen antes de entrenar: proceden de `orientation`. El modelo aprende una relación entre los términos del comentario y esas etiquetas. Los identificadores sirven para agrupar y revisar; no revelan el sentimiento al algoritmo.

Hay tres usos distintos de los datos: **entrenamiento** para aprender; **validación cruzada dentro del entrenamiento** para elegir opciones; y **prueba reservada** para medir el resultado de esa elección. Cada Pipeline aprende su TF-IDF dentro del pliegue correspondiente. Así la prueba y la validación no deciden el vocabulario con el que se entrenan los modelos.

Para exponer las métricas, use un ejemplo real de las salidas: SVM acierta 35 de 75 comentarios, por lo que Accuracy es 46,67 %. Su F1 macro de prueba es 0,3877. La matriz y el informe por clase muestran que neutral se reconoce peor. Estos resultados explican tanto el funcionamiento como los límites de la práctica; la demostración no garantiza que un comentario nuevo se clasifique correctamente.

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
