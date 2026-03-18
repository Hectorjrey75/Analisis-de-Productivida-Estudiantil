"""
Utilidades para manejar rutas del proyecto
Detecta automáticamente la raíz del proyecto
"""

from pathlib import Path
import sys


def get_project_root() -> Path:
    """
    Encuentra la raíz del proyecto buscando carpetas clave
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "config").exists() and (parent / "data").exists():
            return parent

    raise RuntimeError("No se pudo encontrar la raíz del proyecto")


# Definir rutas principales
PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def add_project_to_path():
    """
    Agrega el proyecto al PYTHONPATH para evitar problemas de imports
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))