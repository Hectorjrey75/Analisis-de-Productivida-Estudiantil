"""
Tests unitarios para el módulo src/visualization/visualize.py.
Requisitos: 6.1, 6.2, 6.4
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from src.visualization.visualize import (
    plot_correlation_matrix,
    plot_scatter_with_regression,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
    create_dashboard,
)


# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

@pytest.fixture
def corr_matrix():
    """Matriz de correlación 4x4 sintética."""
    np.random.seed(42)
    x = np.arange(1, 11, dtype=float)
    df = pd.DataFrame({
        'a': x,
        'b': x * 2,
        'c': -x,
        'd': np.random.rand(10),
    })
    return df.corr()


@pytest.fixture
def scatter_df():
    """DataFrame con dos columnas numéricas para scatter plot."""
    np.random.seed(0)
    n = 50
    x = np.random.randn(n)
    return pd.DataFrame({'x': x, 'y': x * 2 + np.random.randn(n) * 0.5})


@pytest.fixture
def importance_df():
    """DataFrame de importancia de features."""
    return pd.DataFrame({
        'feature': [f'feat_{i}' for i in range(15)],
        'importance': np.linspace(0.01, 0.15, 15),
    })


@pytest.fixture
def importance_df_with_ci():
    """DataFrame de importancia con intervalos de confianza."""
    imp = np.linspace(0.01, 0.15, 10)
    return pd.DataFrame({
        'feature': [f'feat_{i}' for i in range(10)],
        'importance': imp,
        'ci_lower': imp - 0.005,
        'ci_upper': imp + 0.005,
    })


@pytest.fixture
def y_arrays():
    """Arrays de valores reales y predichos."""
    np.random.seed(7)
    y_true = np.random.uniform(0, 100, 80)
    y_pred = y_true + np.random.randn(80) * 5
    return y_true, y_pred


@pytest.fixture
def dashboard_results(corr_matrix, importance_df, y_arrays):
    """Diccionario de resultados para create_dashboard."""
    y_true, y_pred = y_arrays
    return {
        'corr_matrix': corr_matrix,
        'feature_importance': importance_df,
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': {'rmse': 5.1, 'mae': 3.8, 'r_squared': 0.82},
    }


# ---------------------------------------------------------------------------
# Tests: plot_correlation_matrix
# ---------------------------------------------------------------------------

class TestPlotCorrelationMatrix:
    def test_runs_without_error(self, corr_matrix):
        """Debe ejecutarse sin lanzar excepciones."""
        plot_correlation_matrix(corr_matrix)

    def test_saves_file(self, corr_matrix, tmp_path):
        """Debe guardar el archivo cuando se proporciona output_path."""
        out = tmp_path / "corr_matrix.png"
        plot_correlation_matrix(corr_matrix, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_file_without_output_path(self, corr_matrix, tmp_path):
        """No debe crear archivo si output_path es None."""
        import os
        before = set(os.listdir(tmp_path))
        plot_correlation_matrix(corr_matrix)
        after = set(os.listdir(tmp_path))
        assert before == after

    def test_accepts_small_matrix(self):
        """Debe funcionar con una matriz 2x2."""
        df = pd.DataFrame({'a': [1.0, 0.5], 'b': [0.5, 1.0]}, index=['a', 'b'])
        plot_correlation_matrix(df)


# ---------------------------------------------------------------------------
# Tests: plot_scatter_with_regression
# ---------------------------------------------------------------------------

class TestPlotScatterWithRegression:
    def test_runs_without_error(self, scatter_df):
        plot_scatter_with_regression(scatter_df, x='x', y='y')

    def test_saves_file(self, scatter_df, tmp_path):
        out = tmp_path / "scatter.png"
        plot_scatter_with_regression(scatter_df, x='x', y='y', output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_file_without_output_path(self, scatter_df, tmp_path):
        import os
        before = set(os.listdir(tmp_path))
        plot_scatter_with_regression(scatter_df, x='x', y='y')
        after = set(os.listdir(tmp_path))
        assert before == after


# ---------------------------------------------------------------------------
# Tests: plot_feature_importance
# ---------------------------------------------------------------------------

class TestPlotFeatureImportance:
    def test_runs_without_error(self, importance_df):
        plot_feature_importance(importance_df)

    def test_saves_file(self, importance_df, tmp_path):
        out = tmp_path / "importance.png"
        plot_feature_importance(importance_df, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_top_n_limits_features(self, importance_df):
        """Debe mostrar solo top_n features sin error."""
        plot_feature_importance(importance_df, top_n=5)

    def test_with_confidence_intervals(self, importance_df_with_ci):
        """Debe funcionar cuando se incluyen intervalos de confianza."""
        plot_feature_importance(importance_df_with_ci)

    def test_raises_on_missing_columns(self):
        """Debe lanzar ValueError si faltan columnas requeridas."""
        bad_df = pd.DataFrame({'name': ['a', 'b'], 'score': [0.1, 0.2]})
        with pytest.raises(ValueError):
            plot_feature_importance(bad_df)


# ---------------------------------------------------------------------------
# Tests: plot_predictions_vs_actual
# ---------------------------------------------------------------------------

class TestPlotPredictionsVsActual:
    def test_runs_without_error(self, y_arrays):
        y_true, y_pred = y_arrays
        plot_predictions_vs_actual(y_true, y_pred)

    def test_saves_file(self, y_arrays, tmp_path):
        y_true, y_pred = y_arrays
        out = tmp_path / "pred_vs_actual.png"
        plot_predictions_vs_actual(y_true, y_pred, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_file_without_output_path(self, y_arrays, tmp_path):
        import os
        y_true, y_pred = y_arrays
        before = set(os.listdir(tmp_path))
        plot_predictions_vs_actual(y_true, y_pred)
        after = set(os.listdir(tmp_path))
        assert before == after

    def test_perfect_predictions(self):
        """Debe funcionar cuando y_true == y_pred."""
        y = np.linspace(0, 10, 20)
        plot_predictions_vs_actual(y, y)


# ---------------------------------------------------------------------------
# Tests: plot_residuals
# ---------------------------------------------------------------------------

class TestPlotResiduals:
    def test_runs_without_error(self, y_arrays):
        y_true, y_pred = y_arrays
        plot_residuals(y_true, y_pred)

    def test_saves_file(self, y_arrays, tmp_path):
        y_true, y_pred = y_arrays
        out = tmp_path / "residuals.png"
        plot_residuals(y_true, y_pred, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_file_without_output_path(self, y_arrays, tmp_path):
        import os
        y_true, y_pred = y_arrays
        before = set(os.listdir(tmp_path))
        plot_residuals(y_true, y_pred)
        after = set(os.listdir(tmp_path))
        assert before == after

    def test_zero_residuals(self):
        """Debe funcionar cuando todos los residuos son cero."""
        y = np.ones(20) * 5.0
        plot_residuals(y, y)


# ---------------------------------------------------------------------------
# Tests: create_dashboard
# ---------------------------------------------------------------------------

class TestCreateDashboard:
    def test_saves_file(self, dashboard_results, tmp_path):
        out = tmp_path / "dashboard.png"
        create_dashboard(dashboard_results, output_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_runs_with_minimal_results(self, tmp_path):
        """Debe funcionar con un diccionario de resultados mínimo."""
        out = tmp_path / "dashboard_minimal.png"
        results = {
            'corr_matrix': pd.DataFrame({'a': [1.0, 0.3], 'b': [0.3, 1.0]}, index=['a', 'b']),
            'feature_importance': pd.DataFrame({'feature': ['f1', 'f2'], 'importance': [0.6, 0.4]}),
            'y_true': np.array([1.0, 2.0, 3.0]),
            'y_pred': np.array([1.1, 1.9, 3.2]),
            'metrics': {'rmse': 0.15, 'mae': 0.13, 'r_squared': 0.98},
        }
        create_dashboard(results, output_path=str(out))
        assert out.exists()

    def test_runs_with_empty_optional_fields(self, tmp_path):
        """Debe funcionar aunque falten campos opcionales en results."""
        out = tmp_path / "dashboard_empty.png"
        results = {
            'corr_matrix': pd.DataFrame(),
            'feature_importance': pd.DataFrame(),
            'y_true': None,
            'y_pred': None,
            'metrics': {},
        }
        create_dashboard(results, output_path=str(out))
        assert out.exists()
