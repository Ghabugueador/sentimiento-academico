# Atribución y licencia de los datos

**Fuente:** Keith, B. (2017). *Paper Reviews* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50G60.

**Página oficial:** https://archive.ics.uci.edu/dataset/410/paper+reviews.

**Licencia del conjunto:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). [Texto legal de la licencia](https://creativecommons.org/licenses/by/4.0/legalcode).

`reviews.json` reproduce el archivo original publicado por UCI, sin modificar su contenido. `articulos_es.csv` es una adaptación de este proyecto: selecciona las evaluaciones en español, conserva el texto y la orientación originales, renombra columnas, mantiene identificadores para auditar y agrupar artículos, deriva el año de la fecha y omite otros campos. Los seis textos vacíos se conservan en el CSV y se excluyen en el notebook.

La clasificación en `negativo`, `neutral` y `positivo` agrupa los valores originales de orientación: −2/−1, 0 y 1/2, respectivamente. Esta agrupación es una decisión metodológica del proyecto, no una nueva anotación de UCI ni de Brian Keith.

Los datos se comparten bajo su licencia original. Al redistribuir los datos o sus adaptaciones, conserve la atribución, el enlace a la licencia y la indicación de los cambios. Esta atribución no implica que el creador del conjunto o UCI aprueben el proyecto o sus resultados.

La licencia aquí descrita corresponde al conjunto de datos y a su adaptación; no declara una licencia para todo el código del proyecto.
