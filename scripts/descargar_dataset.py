#!/usr/bin/env python3
"""Verifica Paper Reviews y deriva el subconjunto español de forma reproducible.

El JSON oficial y su CSV derivado se distribuyen en data/ con atribución CC BY 4.0.
Si falta reviews.json, descarga únicamente el archivo oficial de UCI. No limpia
los textos ni elimina filas vacías: esas exclusiones se explican en el notebook.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


FUENTE_URL = "https://archive.ics.uci.edu/dataset/410/paper+reviews"
DESCARGA_URL = "https://archive.ics.uci.edu/static/public/410/paper%2Breviews.zip"
JSON_SHA256 = "5ea7a36d2655b89db2b73c999f086a3df607089ebd11ebeeb122105d7f0de888"
ZIP_SHA256 = "ee542f0f808921bf599f5678467574a770bb7d3dabac8b10d3dd12d9cf859ec2"
CSV_SHA256 = "12cae78bbc1b73c852c70bde718dabc4ec9bc09969c97e50203ab0707698b81b"
COLUMNAS = [
    "articulo_id", "evaluacion_id", "texto", "orientacion", "idioma", "fecha", "anio"
]


def huella(contenido: bytes) -> str:
    return sha256(contenido).hexdigest()


def comprobar_huella(contenido: bytes, esperada: str, nombre: str) -> None:
    actual = huella(contenido)
    if actual != esperada:
        raise ValueError(
            f"La huella SHA-256 de {nombre} no coincide con la versión documentada. "
            f"Esperada: {esperada}. Encontrada: {actual}. "
            "No se modificaron ni se derivaron datos de ese archivo."
        )


def descargar_json() -> bytes:
    solicitud = Request(DESCARGA_URL, headers={"User-Agent": "PaperReviewsAcademicProject/1.0"})
    with urlopen(solicitud, timeout=60) as respuesta:
        archivo_zip = respuesta.read()
    comprobar_huella(archivo_zip, ZIP_SHA256, "el ZIP oficial")
    with ZipFile(BytesIO(archivo_zip)) as comprimido:
        candidatos = [
            nombre for nombre in comprimido.namelist()
            if Path(nombre).name == "reviews.json" and not nombre.endswith("/")
        ]
        if len(candidatos) != 1:
            raise ValueError("El ZIP oficial no contiene exactamente un reviews.json.")
        contenido = comprimido.read(candidatos[0])
    comprobar_huella(contenido, JSON_SHA256, "reviews.json")
    return contenido


def preparar_csv(contenido: bytes) -> tuple[pd.DataFrame, dict]:
    datos = json.loads(contenido.decode("utf-8"))
    if not isinstance(datos, dict) or not isinstance(datos.get("paper"), list):
        raise ValueError("Estructura inesperada: se requiere una lista en la clave paper.")

    filas = []
    idiomas = Counter()
    ids_articulos = set()
    llaves_evaluaciones = set()
    objetos_sin_evaluaciones = 0
    grupos_totales = set()
    for articulo in datos["paper"]:
        articulo_id = int(articulo["id"])
        if articulo_id in ids_articulos:
            raise ValueError(f"El identificador de artículo {articulo_id} está repetido.")
        ids_articulos.add(articulo_id)
        evaluaciones = articulo["review"]
        if not isinstance(evaluaciones, list):
            raise ValueError("El campo review debe ser una lista.")
        if not evaluaciones:
            objetos_sin_evaluaciones += 1
        for evaluacion in evaluaciones:
            idioma = evaluacion["lan"]
            idiomas[idioma] += 1
            grupos_totales.add(articulo_id)
            evaluacion_id = int(evaluacion["id"])
            llave = (articulo_id, evaluacion_id)
            if llave in llaves_evaluaciones:
                raise ValueError(f"Una evaluación aparece más de una vez: {llave}.")
            llaves_evaluaciones.add(llave)
            if idioma != "es":
                continue
            texto = evaluacion["text"]
            if not isinstance(texto, str):
                raise ValueError("El campo text de una evaluación española no es texto.")
            orientacion = int(evaluacion["orientation"])
            if orientacion not in {-2, -1, 0, 1, 2}:
                raise ValueError(f"Orientación desconocida: {orientacion}.")
            fecha = date.fromisoformat(evaluacion["timespan"])
            filas.append({
                "articulo_id": articulo_id,
                "evaluacion_id": evaluacion_id,
                "texto": texto,
                "orientacion": orientacion,
                "idioma": idioma,
                "fecha": fecha.isoformat(),
                "anio": fecha.year,
            })

    tabla = pd.DataFrame(filas, columns=COLUMNAS)
    no_vacias = tabla[tabla["texto"].str.strip().ne("")]
    etiquetas = no_vacias["orientacion"].map({
        -2: "negativo", -1: "negativo", 0: "neutral", 1: "positivo", 2: "positivo"
    })
    resumen = {
        "objetos_articulo_originales": len(datos["paper"]),
        "articulos_con_evaluaciones": len(grupos_totales),
        "objetos_articulo_sin_evaluaciones": objetos_sin_evaluaciones,
        "evaluaciones_originales": sum(idiomas.values()),
        "idiomas_originales": dict(sorted(idiomas.items())),
        "evaluaciones_espanol_csv": len(tabla),
        "evaluaciones_espanol_texto_vacio": len(tabla) - len(no_vacias),
        "evaluaciones_espanol_texto_no_vacio": len(no_vacias),
        "articulos_espanol_texto_no_vacio": int(no_vacias["articulo_id"].nunique()),
        "orientacion_csv": {
            str(valor): int(cantidad)
            for valor, cantidad in sorted(tabla["orientacion"].value_counts().items())
        },
        "tres_clases_texto_no_vacio": dict(sorted(Counter(etiquetas).items())),
        "anios": sorted(int(valor) for valor in tabla["anio"].unique()),
    }
    return tabla, resumen


def construir_metadatos(csv_bytes: bytes, resumen: dict) -> dict:
    return {
        "dataset": "Paper Reviews",
        "autor": "Brian Keith",
        "anio_publicacion": 2017,
        "repositorio": "UCI Machine Learning Repository",
        "fuente_url": FUENTE_URL,
        "descarga_url": DESCARGA_URL,
        "doi": "10.24432/C50G60",
        "doi_url": "https://doi.org/10.24432/C50G60",
        "licencia": "CC BY 4.0",
        "licencia_url": "https://creativecommons.org/licenses/by/4.0/",
        "archivo_original": "reviews.json",
        "archivo_derivado": "articulos_es.csv",
        "sha256_json_original": JSON_SHA256,
        "sha256_zip_oficial": ZIP_SHA256,
        "sha256_csv_derivado": huella(csv_bytes),
        "version_transformacion": 1,
        "codificacion_csv": "UTF-8 sin BOM; terminador de línea LF",
        "columnas_csv": COLUMNAS,
        "transformacion": [
            "Recorrer paper y review en su orden original.",
            "Seleccionar solamente evaluaciones con lan igual a es.",
            "Conservar text íntegro, incluido el texto vacío, sin traducir ni sintetizar.",
            "Renombrar id de paper como articulo_id e id de review como evaluacion_id.",
            "Renombrar text, orientation y lan como texto, orientacion e idioma.",
            "Conservar timespan como fecha y derivar anio de esa fecha.",
            "Omitir remarks, preliminary_decision, evaluation y confidence del CSV.",
        ],
        "clave_evaluacion": ["articulo_id", "evaluacion_id"],
        "grupo_particion": "articulo_id",
        "variable_predictora": "texto",
        "objetivo_original": "orientation (percepción del sentimiento del texto)",
        "mapeo_objetivo_tres_clases": {
            "-2": "negativo", "-1": "negativo", "0": "neutral",
            "1": "positivo", "2": "positivo",
        },
        "nota_objetivo": (
            "La agrupación en tres clases es una adaptación de este proyecto. "
            "Orientation no es evaluation ni la decisión editorial de aceptación."
        ),
        "nota_modelo": (
            "Los identificadores, fecha, año, idioma y orientación no se usan como "
            "características predictoras. Los identificadores sirven para auditoría "
            "y para mantener las evaluaciones de un artículo en una misma partición."
        ),
        "estadisticas": resumen,
        "atribucion": (
            "Keith, B. (2017). Paper Reviews [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C50G60. CC BY 4.0. "
            "articulos_es.csv es un subconjunto derivado en español."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Carpeta de datos; por defecto, data/ del proyecto.",
    )
    parser.add_argument(
        "--verificar", action="store_true",
        help="Verificar archivos existentes sin descargar ni escribir archivos.",
    )
    argumentos = parser.parse_args()
    carpeta = argumentos.data_dir.resolve()
    archivo_json = carpeta / "reviews.json"
    archivo_csv = carpeta / "articulos_es.csv"
    archivo_metadatos = carpeta / "source.json"

    if archivo_json.is_file():
        contenido = archivo_json.read_bytes()
        comprobar_huella(contenido, JSON_SHA256, "reviews.json")
    elif argumentos.verificar:
        raise FileNotFoundError(f"Falta el archivo original: {archivo_json}")
    else:
        contenido = descargar_json()
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo_json.write_bytes(contenido)

    tabla, resumen = preparar_csv(contenido)
    csv_bytes = tabla.to_csv(index=False, lineterminator="\n").encode("utf-8")
    comprobar_huella(csv_bytes, CSV_SHA256, "el CSV derivado")
    metadatos = construir_metadatos(csv_bytes, resumen)
    metadatos_bytes = (json.dumps(metadatos, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    if argumentos.verificar:
        comprobar_huella(archivo_csv.read_bytes(), CSV_SHA256, "articulos_es.csv")
        if archivo_metadatos.read_bytes() != metadatos_bytes:
            raise ValueError("source.json no coincide con la transformación documentada.")
    else:
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo_csv.write_bytes(csv_bytes)
        archivo_metadatos.write_bytes(metadatos_bytes)

    print("Paper Reviews verificado: 405 evaluaciones originales.")
    print(
        f"CSV español: {resumen['evaluaciones_espanol_csv']} evaluaciones; "
        f"{resumen['evaluaciones_espanol_texto_vacio']} con texto vacío."
    )
    print(f"SHA-256 CSV: {huella(csv_bytes)}")
    print("El notebook aplica las exclusiones y utiliza únicamente texto para predecir.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error de verificación: {error}") from error
