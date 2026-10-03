"""Ejecuta todas las celdas con este Python y guarda salidas y vista HTML."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks" / "sentiment_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

with TemporaryDirectory(prefix="sentiment-kernel-") as temporary:
    kernel_root = Path(temporary)
    specification = kernel_root / "sentiment-local"
    specification.mkdir()
    (specification / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Sentiment (Python local)", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(
        kernel_name="sentiment-local",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )
    def progress(cell, cell_index):
        if cell.cell_type == "code":
            print(f"Ejecutando celda de código: {cell.source.splitlines()[0]}", flush=True)
    client = NotebookClient(
        notebook, km=manager, timeout=3600, allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_start=progress,
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=False)

nbformat.validate(notebook)
nbformat.write(notebook, path)
html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
path.with_suffix(".html").write_text(html, encoding="utf-8")
print(f"Notebook ejecutado y guardado: {path}")
