"""
Tests de integración end-to-end para pipeline.py.

"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Asegurar que el root del proyecto está en sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import (
    ALL_STAGES,
    build_parser,
    configure_logging,
    load_config,
    run_pipeline,
    stage_correlation,
    stage_data,
    stage_export,
    stage_features,
    stage_preprocess,
    stage_recommend,
    stage_train,
    stage_visualize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """DataFrame  con las columnas requeridas por el pipeline."""
    rng = np.random.default_rng(42)
    n = 80

    df = pd.DataFrame(
        {
            "student_id": range(1, n + 1),
            "productivity_score": rng.uniform(30, 100, n),
            "final_grade": rng.uniform(40, 100, n),
            "study_hours_per_day": rng.uniform(1, 10, n),
            "phone_usage_hours": rng.uniform(0, 8, n),
            "social_media_hours": rng.uniform(0, 5, n),
            "youtube_hours": rng.uniform(0, 4, n),
            "gaming_hours": rng.uniform(0, 3, n),
            "sleep_hours": rng.uniform(4, 10, n),
            "exercise_minutes": rng.uniform(0, 120, n),
            "attendance_percentage": rng.uniform(50, 100, n),
            "assignments_completed": rng.integers(0, 20, n).astype(float),
        }
    )
    return df


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Configuración mínima para tests."""
    return {
        "project": {"name": "test", "version": "0.0.1"},
        "data": {
            "raw_path": "data/raw/student_productivity_distraction_dataset_20000.csv",
            "processed_path": "data/processed/",
            "required_columns": [
                "student_id",
                "productivity_score",
                "final_grade",
                "study_hours_per_day",
                "phone_usage_hours",
                "social_media_hours",
                "youtube_hours",
                "gaming_hours",
                "sleep_hours",
                "exercise_minutes",
                "attendance_percentage",
                "assignments_completed",
            ],
            "imputation": {"strategy": "median"},
        },
        "features": {
            "digital": ["phone_usage_hours", "social_media_hours", "youtube_hours", "gaming_hours"],
            "academic": ["study_hours_per_day", "attendance_percentage", "assignments_completed", "final_grade"],
            "lifestyle": ["sleep_hours", "exercise_minutes"],
            "target": ["productivity_score", "final_grade"],
        },
        "preprocessing": {
            "normalization": {"method": "standard"},
            "encoding": {"method": "onehot"},
            "outliers": {"strategy": "clip", "threshold": 3.0},
        },
        "correlation": {"method": "pearson", "significance_level": 0.95},
        "models": {
            "types": ["linear_regression", "random_forest"],
            "regression": {
                "linear_regression": {},
                "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
                "gradient_boosting": {},
            },
            "production_threshold": {"r_squared": 0.0},
        },
        "training": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "cv_folds": 3,
        },
        "recommendations": {
            "n_recommendations": 3,
            "optimal_ranges": {
                "study_hours_per_day": [6.0, 9.0],
                "sleep_hours": [7.0, 9.0],
                "phone_usage_hours": [0.0, 3.0],
                "social_media_hours": [0.0, 2.0],
                "exercise_minutes": [30, 60],
            },
        },
        "export": {"model_formats": ["pickle"], "include_metadata": True, "compute_checksums": True},
        "visualization": {"style": "seaborn-v0_8-darkgrid", "palette": "husl", "figsize": [12, 8], "dpi": 72},
    }


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------

def test_load_config_reads_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("project:\n  name: test\n  version: '1.0'\n")
    config = load_config(str(cfg_file))
    assert config["project"]["name"] == "test"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


# ---------------------------------------------------------------------------
# Tests: individual stages
# ---------------------------------------------------------------------------

def test_stage_features(synthetic_df, minimal_config):
    logger = configure_logging()
    result = stage_features(synthetic_df, logger)
    # Derived features should be added
    assert "total_screen_time" in result.columns
    assert "study_sleep_ratio" in result.columns
    assert len(result) == len(synthetic_df)


def test_stage_correlation(synthetic_df, minimal_config):
    logger = configure_logging()
    corr_matrix, sig_corr = stage_correlation(synthetic_df, minimal_config, logger)
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    assert "productivity_score" in corr_matrix.columns


def test_stage_preprocess(synthetic_df, minimal_config):
    logger = configure_logging()
    df_processed, preprocessor = stage_preprocess(synthetic_df, minimal_config, logger)
    assert df_processed.shape[0] == synthetic_df.shape[0]
    assert preprocessor is not None


def test_stage_train(synthetic_df, minimal_config):
    logger = configure_logging()
    model, importance_df, X_test, y_test, metrics, feature_cols = stage_train(
        synthetic_df, minimal_config, logger
    )
    assert model is not None
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert len(metrics) > 0
    for m in metrics.values():
        assert "r_squared" in m
        assert "rmse" in m


def test_stage_recommend(synthetic_df, minimal_config):
    logger = configure_logging()
    model, importance_df, X_test, y_test, metrics, feature_cols = stage_train(
        synthetic_df, minimal_config, logger
    )
    recommendations = stage_recommend(
        synthetic_df, model, importance_df, minimal_config, logger, feature_cols=feature_cols
    )
    assert isinstance(recommendations, list)


def test_stage_export(synthetic_df, minimal_config, tmp_path):
    logger = configure_logging()
    model, importance_df, X_test, y_test, metrics, feature_cols = stage_train(
        synthetic_df, minimal_config, logger
    )
    best_metrics = max(metrics.values(), key=lambda m: m.get("r_squared", -1))
    stage_export(synthetic_df, model, [], best_metrics, minimal_config, str(tmp_path), logger)
    assert (tmp_path / "processed_data.csv").exists()
    assert (tmp_path / "results.json").exists()


# ---------------------------------------------------------------------------
# Tests: run_pipeline
# ---------------------------------------------------------------------------

def test_run_pipeline_all_stages(synthetic_df, minimal_config, tmp_path, monkeypatch):
    """Pipeline completo con datos sintéticos."""
    # Patch stage_data to return synthetic_df instead of reading from disk
    import pipeline as pl
    monkeypatch.setattr(pl, "stage_data", lambda config, logger: synthetic_df)

    artifacts = run_pipeline(minimal_config, stages=ALL_STAGES, output_dir=str(tmp_path))

    assert "df_raw" in artifacts
    assert "df_processed" in artifacts
    assert "df_features" in artifacts
    assert "corr_matrix" in artifacts
    assert "model" in artifacts
    assert "importance_df" in artifacts
    assert "recommendations" in artifacts


def test_run_pipeline_selective_stages(synthetic_df, minimal_config, tmp_path, monkeypatch):
    """Pipeline con etapas seleccionadas."""
    import pipeline as pl
    monkeypatch.setattr(pl, "stage_data", lambda config, logger: synthetic_df)

    artifacts = run_pipeline(
        minimal_config,
        stages=["data", "preprocess", "features"],
        output_dir=str(tmp_path),
    )

    assert "df_raw" in artifacts
    assert "df_processed" in artifacts
    assert "df_features" in artifacts
    # Stages not run should not be in artifacts
    assert "model" not in artifacts
    assert "corr_matrix" not in artifacts


def test_run_pipeline_train_only(synthetic_df, minimal_config, tmp_path):
    """Etapa train con DataFrame ya disponible en artifacts."""
    import pipeline as pl

    logger = configure_logging()
    # Manually build artifacts up to features
    df_feat = stage_features(synthetic_df, logger)

    # Inject into run_pipeline by patching stage_data and stage_preprocess/features
    import unittest.mock as mock

    with mock.patch.object(pl, "stage_data", return_value=synthetic_df), \
         mock.patch.object(pl, "stage_preprocess", return_value=(synthetic_df, None)), \
         mock.patch.object(pl, "stage_features", return_value=df_feat):
        artifacts = run_pipeline(
            minimal_config,
            stages=["data", "preprocess", "features", "train"],
            output_dir=str(tmp_path),
        )

    assert "model" in artifacts
    assert "importance_df" in artifacts


def test_run_pipeline_export_creates_files(synthetic_df, minimal_config, tmp_path, monkeypatch):
    """Verifica que la etapa export crea los archivos esperados."""
    import pipeline as pl
    monkeypatch.setattr(pl, "stage_data", lambda config, logger: synthetic_df)

    run_pipeline(
        minimal_config,
        stages=["data", "preprocess", "features", "train", "export"],
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "processed_data.csv").exists()
    assert (tmp_path / "results.json").exists()


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------

def test_cli_default_args():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == "config/config.yaml"
    assert args.stages is None
    assert args.output_dir == "data/processed"


def test_cli_custom_config():
    parser = build_parser()
    args = parser.parse_args(["--config", "my_config.yaml"])
    assert args.config == "my_config.yaml"


def test_cli_stages_parsed():
    parser = build_parser()
    args = parser.parse_args(["--stages", "data,preprocess,train"])
    assert args.stages == "data,preprocess,train"


def test_cli_output_dir():
    parser = build_parser()
    args = parser.parse_args(["--output-dir", "/tmp/results"])
    assert args.output_dir == "/tmp/results"


def test_cli_invalid_stage_returns_error(tmp_path, minimal_config):
    """CLI debe retornar código de error 1 para etapas inválidas."""
    import pipeline as pl

    # Write a valid config file
    import yaml
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(minimal_config, f)

    result = pl.main(["--config", str(cfg_path), "--stages", "invalid_stage"])
    assert result == 1


def test_cli_missing_config_returns_error():
    """CLI debe retornar código de error 1 si el config no existe."""
    import pipeline as pl
    result = pl.main(["--config", "nonexistent.yaml"])
    assert result == 1
