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
