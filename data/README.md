# Datos: evaluaciones de artículos científicos en español

Este proyecto utiliza **Paper Reviews**, publicado por Brian Keith en UCI Machine Learning Repository. El corpus contiene evaluaciones de artículos enviados a una conferencia de computación e informática. Los textos son originales del corpus; no se tradujeron ni se generaron comentarios sintéticos.

Fuente oficial: [Paper Reviews en UCI](https://archive.ics.uci.edu/dataset/410/paper+reviews). DOI: [10.24432/C50G60](https://doi.org/10.24432/C50G60). Licencia: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). La atribución y los cambios de este proyecto están en [ATTRIBUTION.md](ATTRIBUTION.md).

## Archivos

- `reviews.json`: archivo original de UCI, sin modificaciones. Contiene 405 evaluaciones: 388 en español y 17 en inglés.
- `articulos_es.csv`: subconjunto derivado de las 388 evaluaciones con `lan == "es"`, conservadas en el orden original. Incluye los seis textos vacíos para que el notebook muestre y justifique su exclusión.
- `source.json`: procedencia, licencia, transformación, recuentos y huellas SHA-256.

Los archivos pequeños se incluyen en el proyecto para ejecutarlo sin descargar nuevamente los datos. Si falta `reviews.json`, `scripts/descargar_dataset.py` descarga el ZIP oficial y comprueba sus huellas antes de derivar el CSV. No se necesita una cuenta ni una clave de API.

## Columnas del CSV

| Columna | Procedencia y función |
|---|---|
| `articulo_id` | `paper.id` original. Sirve para mantener todas las evaluaciones de un mismo artículo en una misma partición. No es una característica del modelo. |
| `evaluacion_id` | `review.id` original; es local al artículo. La pareja `articulo_id`, `evaluacion_id` identifica una evaluación. |
| `texto` | Campo `text` íntegro de la evaluación. Es la única entrada predictora de los modelos. |
| `orientacion` | Campo `orientation`, entero entre −2 y 2. Se utiliza exclusivamente para construir la etiqueta objetivo. |
| `idioma` | Campo `lan`, siempre `es` en este subconjunto. Es información de auditoría. |
| `fecha` | Campo `timespan` de la evaluación, en formato ISO. Es información de auditoría. |
| `anio` | Año obtenido de `fecha`. El subconjunto incluye 2010, 2013, 2014 y 2015. |

`remarks`, `preliminary_decision`, `evaluation` y `confidence` no se trasladan al CSV ni se utilizan para predecir. El modelo tampoco recibe identificadores, idioma, fecha o año.

## Objetivo y exclusiones

El objetivo es predecir la orientación sentimental percibida al leer la evaluación. UCI distingue `orientation`, asignada por los autores del estudio mediante su juicio sistemático, de `evaluation`, que es la valoración del evaluador. También distingue ambas de la decisión editorial de aceptación o rechazo del artículo.

Este proyecto adapta la escala original de cinco valores a tres clases:

| Orientación original | Etiqueta del proyecto |
|---|---|
| −2 y −1 | `negativo` |
| 0 | `neutral` |
| 1 y 2 | `positivo` |

Los seis textos vacíos tienen orientación 0. Tras excluirlos quedan **382 evaluaciones de 168 artículos**: 172 negativas, 104 neutrales y 106 positivas. Las 17 evaluaciones en inglés se excluyen durante la creación del CSV. No se sustituyen textos vacíos ni se inventan etiquetas.

El JSON original tiene 172 objetos de artículo. Tres objetos no contienen evaluaciones; por eso hay 169 artículos con evaluaciones en el corpus completo. Estos números describen niveles distintos del conjunto, no una pérdida accidental de filas.

El corpus es pequeño, antiguo y de un dominio específico. Los resultados describen este experimento; no permiten inferir aceptación editorial, calidad científica ni rendimiento general en otras conferencias.

## Reproducción

Desde la carpeta principal del proyecto, con sus dependencias instaladas:

```console
python scripts/descargar_dataset.py
python scripts/descargar_dataset.py --verificar
```

El primer comando verifica el JSON disponible y produce el CSV y sus metadatos. El segundo comprueba los archivos existentes sin descargarlos ni modificarlos. Si una huella cambia, el programa se detiene con un mensaje; no acepta automáticamente una versión distinta.

El CSV utiliza UTF-8 sin BOM y terminadores de línea LF. La transformación mantiene los textos originales, conserva el orden y no depende de un muestreo aleatorio.

| Archivo | SHA-256 |
|---|---|
| `reviews.json` | `5ea7a36d2655b89db2b73c999f086a3df607089ebd11ebeeb122105d7f0de888` |
| `articulos_es.csv` | `12cae78bbc1b73c852c70bde718dabc4ec9bc09969c97e50203ab0707698b81b` |
| ZIP oficial | `ee542f0f808921bf599f5678467574a770bb7d3dabac8b10d3dd12d9cf859ec2` |
