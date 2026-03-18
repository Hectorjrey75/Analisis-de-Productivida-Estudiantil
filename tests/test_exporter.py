"""Tests unitarios para el módulo de exportación."""

import hashlib
import json
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.export.exporter import (
    compute_checksum,
    export_dataframe,
    export_model,
    export_results_with_metadata,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4.0, 5.0, 6.0],
        'c': ['x', 'y', 'z'],
    })


@pytest.fixture
def trained_model():
    X = [[1], [2], [3]]
    y = [2, 4, 6]
    model = LinearRegression()
    model.fit(X, y)
    return model


# --- export_dataframe ---

def test_export_dataframe_csv(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'out.csv')
        export_dataframe(sample_df, path, format='csv')
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == list(sample_df.columns)
        assert len(loaded) == len(sample_df)


def test_export_dataframe_json(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'out.json')
        export_dataframe(sample_df, path, format='json')
        loaded = pd.read_json(path, orient='records')
        assert list(loaded.columns) == list(sample_df.columns)
        assert len(loaded) == len(sample_df)


@pytest.mark.skipif(
    not any(__import__('importlib').util.find_spec(pkg) for pkg in ('pyarrow', 'fastparquet')),
    reason="pyarrow or fastparquet required for parquet support",
)
def test_export_dataframe_parquet(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'out.parquet')
        export_dataframe(sample_df, path, format='parquet')
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(sample_df.columns)
        assert len(loaded) == len(sample_df)


def test_export_dataframe_invalid_format(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'out.xyz')
        with pytest.raises(ValueError, match="Unsupported format"):
            export_dataframe(sample_df, path, format='xyz')


def test_export_dataframe_creates_parent_dirs(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'nested' / 'dir' / 'out.csv')
        export_dataframe(sample_df, path, format='csv')
        assert Path(path).exists()


# --- export_model ---

def test_export_model_pickle(trained_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'model')
        export_model(trained_model, path, formats=['pickle'])
        pkl_path = Path(tmpdir) / 'model.pkl'
        assert pkl_path.exists()
        with open(pkl_path, 'rb') as f:
            loaded = pickle.load(f)
        assert hasattr(loaded, 'predict')


def test_export_model_onnx_graceful_fallback(trained_model):
    """ONNX export should not raise even if skl2onnx is unavailable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'model')
        # Should not raise regardless of whether skl2onnx is installed
        export_model(trained_model, path, formats=['onnx'])


def test_export_model_multiple_formats(trained_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'model')
        export_model(trained_model, path, formats=['pickle', 'onnx'])
        assert (Path(tmpdir) / 'model.pkl').exists()


# --- export_results_with_metadata ---

def test_export_results_with_metadata_creates_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'results.json')
        results = {'rmse': 0.5, 'r2': 0.85}
        metadata = {'version': '1.0.0', 'config': {'model': 'rf'}}
        export_results_with_metadata(results, path, metadata)
        assert Path(path).exists()


def test_export_results_with_metadata_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / 'results.json')
        results = {'rmse': 0.5, 'r2': 0.85}
        metadata = {'version': '1.0.0'}
        export_results_with_metadata(results, path, metadata)
        with open(path, 'r') as f:
            data = json.load(f)
        assert 'results' in data
        assert 'metadata' in data
        assert data['results']['rmse'] == 0.5
        assert 'date' in data['metadata']
        assert data['metadata']['version'] == '1.0.0'


# --- compute_checksum ---

def test_compute_checksum_returns_hex_string():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(b'hello world')
        tmp_path = f.name
    checksum = compute_checksum(tmp_path)
    assert isinstance(checksum, str)
    assert len(checksum) == 64  # SHA256 hex digest length


def test_compute_checksum_correct_value():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(b'hello world')
        tmp_path = f.name
    expected = hashlib.sha256(b'hello world').hexdigest()
    assert compute_checksum(tmp_path) == expected


def test_compute_checksum_different_files_differ():
    with tempfile.NamedTemporaryFile(delete=False) as f1:
        f1.write(b'content A')
        path1 = f1.name
    with tempfile.NamedTemporaryFile(delete=False) as f2:
        f2.write(b'content B')
        path2 = f2.name
    assert compute_checksum(path1) != compute_checksum(path2)
