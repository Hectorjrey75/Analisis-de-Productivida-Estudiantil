import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    """Carga la configuración desde archivo YAML"""
    
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# Configuración global
CONFIG = load_config()

PROJECT_ROOT = Path(__file__).parent.parent