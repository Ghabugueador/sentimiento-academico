@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primero cree el entorno Python siguiendo README.md, apartado 4.5.
    echo py -3.12 -m venv .venv
    echo .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab "notebooks/sentiment_analysis.ipynb" %*
exit /b %ERRORLEVEL%
