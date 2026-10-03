@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primero cree el entorno Python siguiendo README.md, apartado 4.5.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab "notebooks/sentiment_analysis.ipynb"
