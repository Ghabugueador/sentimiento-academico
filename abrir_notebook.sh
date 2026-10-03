#!/bin/sh
# Abre el mismo notebook en Linux con el entorno creado en este equipo.
set -eu

case "$0" in
    */*) DIRECTORIO_SCRIPT=${0%/*} ;;
    *) DIRECTORIO_SCRIPT=. ;;
esac
RAIZ_PROYECTO=$(CDPATH= cd -- "$DIRECTORIO_SCRIPT" && pwd)
cd "$RAIZ_PROYECTO"

PYTHON_PROYECTO="$RAIZ_PROYECTO/.venv/bin/python"
if [ ! -x "$PYTHON_PROYECTO" ]; then
    printf '%s\n' \
        'Primero cree el entorno Python siguiendo README.md, apartado 4.5.' \
        'python3.12 -m venv .venv' \
        '.venv/bin/python -m pip install -r requirements-lock.txt'
    exit 1
fi

exec "$PYTHON_PROYECTO" -m jupyterlab "notebooks/sentiment_analysis.ipynb" "$@"
