"""Comprueba la procedencia, separación y coherencia del experimento guardado.

No entrena ni modifica archivos. Ejecútelo después de ejecutar el notebook.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASES = ["negativo", "neutral", "positivo"]


def limpiar_texto(texto):
    texto = unicodedata.normalize("NFC", html.unescape(str(texto))).lower()
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = "".join(c if c.isalpha() or c.isspace() else " " for c in texto)
    return re.sub(r"\s+", " ", texto).strip()


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/descargar_dataset.py"), "--verificar"], check=True)
    notebook = nbformat.read(ROOT / "notebooks/sentiment_analysis.ipynb", as_version=4)
    nbformat.validate(notebook)
    celdas = [celda for celda in notebook.cells if celda.cell_type == "code"]
    assert len(celdas) == 23, "Se esperan 23 celdas de código."
    assert [c.execution_count for c in celdas] == list(range(1, 24)), "Ejecute el notebook completo en orden."
    assert not any(o.output_type == "error" for c in celdas for o in c.outputs), "Hay errores guardados."
    assert "�" not in json.dumps(notebook, ensure_ascii=False), "Revise la codificación de las salidas."
    try:
        exec(celdas[-1].source, {})
    except RuntimeError as error:
        assert "Primero ejecute todas las celdas" in str(error)
    else:
        raise AssertionError("La demo debe indicar que falta el entrenamiento en un kernel nuevo.")

    ruta_datos = ROOT / "data/articulos_es.csv"
    datos = pd.read_csv(ruta_datos, keep_default_na=False)
    datos = datos.loc[datos.texto.str.strip().ne("")].copy()
    datos = datos.drop_duplicates(["texto", "orientacion"])
    datos["texto_limpio"] = datos.texto.map(limpiar_texto)
    datos = datos.loc[datos.texto_limpio.ne("")].drop_duplicates("texto_limpio").reset_index(drop=True)
    datos["real"] = datos.orientacion.map({-2: "negativo", -1: "negativo", 0: "neutral", 1: "positivo", 2: "positivo"})
    resultados = ROOT / "resultados"
    metricas = json.loads((resultados / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["sha256_dataset"] == hashlib.sha256(ruta_datos.read_bytes()).hexdigest()
    indices = pd.read_csv(resultados / "indices_particion.csv")
    assert set(indices.fila_modelado) == set(datos.index)
    assert not indices.fila_modelado.duplicated().any()
    origen = datos[["articulo_id", "evaluacion_id"]].reset_index(names="fila_modelado")
    assert indices[["fila_modelado", "articulo_id", "evaluacion_id"]].equals(origen)
    train = indices.loc[indices.conjunto.eq("entrenamiento")]
    test = indices.loc[indices.conjunto.eq("prueba")]
    assert len(train) + len(test) == len(datos)
    assert set(train.articulo_id).isdisjoint(test.articulo_id), "Artículos compartidos."
    assert set(datos.loc[train.fila_modelado, "texto_limpio"]).isdisjoint(datos.loc[test.fila_modelado, "texto_limpio"])
    assert len(train) == metricas["entrenamiento"] and len(test) == metricas["prueba"]
    assert train.articulo_id.nunique() == metricas["articulos_entrenamiento"]
    assert test.articulo_id.nunique() == metricas["articulos_prueba"]
    predicciones = pd.read_csv(resultados / "predicciones_prueba.csv")
    assert len(predicciones) == len(test)
    assert set(predicciones.fila_modelado) == set(test.fila_modelado)
    assert predicciones.real.tolist() == datos.loc[predicciones.fila_modelado, "real"].tolist()
    assert set(predicciones.predicho).issubset(CLASES)
    ganador = max(metricas["resultados_cv"], key=lambda fila: fila["F1_macro_CV"])["Modelo"]
    assert ganador == metricas["modelo_seleccionado"]
    fila = next(fila for fila in metricas["resultados_prueba"] if fila["Modelo"] == ganador)
    assert np.isclose(accuracy_score(predicciones.real, predicciones.predicho), fila["Accuracy"])
    assert np.isclose(f1_score(predicciones.real, predicciones.predicho, labels=CLASES, average="macro"), fila["F1_macro"])
    matriz = pd.read_csv(resultados / "matriz_confusion.csv", index_col=0).loc[CLASES, CLASES].to_numpy()
    assert np.array_equal(matriz, confusion_matrix(predicciones.real, predicciones.predicho, labels=CLASES))
    print("Comprobación correcta: 23 celdas, datos verificados y artículos separados.")
    print("Etiquetas, predicciones, métricas y matriz de confusión coinciden.")


if __name__ == "__main__":
    main()
