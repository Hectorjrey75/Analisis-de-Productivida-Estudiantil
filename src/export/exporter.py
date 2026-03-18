"""Módulo de exportación de datos, modelos y resultados."""

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def export_dataframe(df: pd.DataFrame, output_path: str, format: str = 'csv') -> None:
    """Exports DataFrame in specified format.

    Args:
        df: DataFrame to export.
        output_path: Destination file path.
        format: Output format - 'csv', 'json', or 'parquet'.

    Raises:
        ValueError: If format is not supported.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'json':
        df.to_json(path, orient='records', indent=2)
    elif format == 'parquet':
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'csv', 'json', or 'parquet'.")


def export_model(model: Any, output_path: str, formats: List[str] = ['pickle']) -> None:
    """Exports model in multiple formats.

    Args:
        model: Trained scikit-learn model to export.
        output_path: Base destination path (extension will be added per format).
        formats: List of formats to export - 'pickle' and/or 'onnx'.
    """
    base_path = Path(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt == 'pickle':
            pickle_path = base_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)

        elif fmt == 'onnx':
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                # Determine input shape from model if possible
                n_features = None
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_

                initial_type = [('float_input', FloatTensorType([None, n_features]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)

                onnx_path = base_path.with_suffix('.onnx')
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())

            except ImportError:
                # skl2onnx not installed - skip ONNX export gracefully
                pass


def export_results_with_metadata(results: Dict, output_path: str, metadata: Dict) -> None:
    """Exports results with metadata (date, version, config).

    Args:
        results: Dictionary of results to export.
        output_path: Destination JSON file path.
        metadata: Additional metadata to include (version, config, etc.).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        'metadata': {
            'date': datetime.now().isoformat(),
            **metadata,
        },
        'results': results,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, default=str)


def compute_checksum(file_path: str) -> str:
    """Calculates SHA256 checksum of a file.

    Args:
        file_path: Path to the file to checksum.

    Returns:
        SHA256 hash as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
