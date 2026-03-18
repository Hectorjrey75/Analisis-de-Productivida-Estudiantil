"""
Tests unitarios para src/models/monitor.py.
Cubre ModelMonitor: constructor, evaluate_and_monitor, get_performance_history.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.monitor import ModelMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor():
    return ModelMonitor()


@pytest.fixture
def synthetic_data():
    """Dataset sintético con relación lineal fuerte."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"a": np.random.uniform(1, 10, n), "b": np.random.uniform(0, 5, n)})
    y = pd.Series(3 * X["a"] + 2 * X["b"] + np.random.randn(n) * 0.3, name="target")
    return X, y


@pytest.fixture
def trained_model(synthetic_data):
    X, y = synthetic_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestModelMonitorConstructor:
    def test_default_threshold(self):
        m = ModelMonitor()
        assert m.degradation_threshold == 0.10

    def test_custom_threshold(self):
        m = ModelMonitor(degradation_threshold=0.20)
        assert m.degradation_threshold == 0.20

    def test_history_initialized_empty(self):
        m = ModelMonitor()
        assert isinstance(m.history, list)
        assert len(m.history) == 0


# ---------------------------------------------------------------------------
# evaluate_and_monitor — métricas correctas
# ---------------------------------------------------------------------------

class TestEvaluateAndMonitor:
    def test_returns_dict_with_required_keys(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert "rmse" in result
        assert "mae" in result
        assert "r_squared" in result
        assert "model_name" in result
        assert "timestamp" in result

    def test_model_name_stored(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "my_model")
        assert result["model_name"] == "my_model"

    def test_rmse_non_negative(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert result["rmse"] >= 0.0

    def test_mae_non_negative(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert result["mae"] >= 0.0

    def test_r_squared_at_most_one(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert result["r_squared"] <= 1.0

    def test_rmse_gte_mae(self, monitor, trained_model):
        """RMSE >= MAE por desigualdad de Jensen."""
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert result["rmse"] >= result["mae"]

    def test_entry_appended_to_history(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        assert len(monitor.history) == 1

    def test_multiple_calls_append_multiple_entries(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        monitor.evaluate_and_monitor(model, X, y, "lr")
        assert len(monitor.history) == 2

    def test_timestamp_is_string(self, monitor, trained_model):
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert isinstance(result["timestamp"], str)
        # ISO format check: contains 'T'
        assert "T" in result["timestamp"]

    def test_high_r2_for_linear_data(self, monitor, trained_model):
        """Modelo lineal sobre datos lineales debe tener R² alto."""
        model, X, y = trained_model
        result = monitor.evaluate_and_monitor(model, X, y, "lr")
        assert result["r_squared"] > 0.9


# ---------------------------------------------------------------------------
# evaluate_and_monitor — detección de degradación
# ---------------------------------------------------------------------------

class TestDegradationDetection:
    def test_no_warning_on_first_evaluation(self, monitor, trained_model):
        """Primera evaluación no debe emitir alerta (sin historial previo)."""
        model, X, y = trained_model
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.evaluate_and_monitor(model, X, y, "lr")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_when_performance_stable(self, monitor, trained_model):
        """Sin degradación significativa → sin alerta."""
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.evaluate_and_monitor(model, X, y, "lr")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_warning_emitted_when_r2_drops_significantly(self, monitor):
        """Alerta cuando R² cae más del umbral configurado."""
        np.random.seed(0)
        n = 100
        X = pd.DataFrame({"a": np.random.randn(n)})

        # Primer modelo: buenas predicciones (R² alto)
        y_good = pd.Series(5 * X["a"].values + np.random.randn(n) * 0.1)
        good_model = LinearRegression()
        good_model.fit(X, y_good)
        monitor.evaluate_and_monitor(good_model, X, y_good, "model_x")

        # Segundo modelo: predicciones malas (R² bajo) para el mismo nombre
        y_bad = pd.Series(np.random.randn(n) * 10)
        bad_model = LinearRegression()
        bad_model.fit(X, y_bad)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.evaluate_and_monitor(bad_model, X, y_bad, "model_x")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert "model_x" in str(user_warnings[0].message)

    def test_warning_message_contains_model_name(self, monitor):
        """El mensaje de alerta debe incluir el nombre del modelo."""
        np.random.seed(1)
        n = 80
        X = pd.DataFrame({"f": np.random.randn(n)})
        y_good = pd.Series(10 * X["f"].values)
        good_model = LinearRegression()
        good_model.fit(X, y_good)
        monitor.evaluate_and_monitor(good_model, X, y_good, "my_special_model")

        y_bad = pd.Series(np.random.randn(n) * 50)
        bad_model = LinearRegression()
        bad_model.fit(X, y_bad)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.evaluate_and_monitor(bad_model, X, y_bad, "my_special_model")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("my_special_model" in str(warning.message) for warning in user_warnings)

    def test_no_cross_model_degradation_check(self, monitor, trained_model):
        """Degradación solo se compara contra el mismo model_name."""
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "model_a")

        # Modelo malo evaluado con nombre diferente → sin alerta
        np.random.seed(99)
        X2 = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y2 = pd.Series(np.random.randn(50) * 100)
        bad_model = LinearRegression()
        bad_model.fit(X2, y2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.evaluate_and_monitor(bad_model, X2, y2, "model_b")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_custom_threshold_respected(self):
        """Umbral personalizado debe usarse en la detección."""
        monitor_strict = ModelMonitor(degradation_threshold=0.01)
        np.random.seed(5)
        n = 100
        X = pd.DataFrame({"x": np.random.randn(n)})
        y_good = pd.Series(3 * X["x"].values + np.random.randn(n) * 0.1)
        good_model = LinearRegression()
        good_model.fit(X, y_good)
        monitor_strict.evaluate_and_monitor(good_model, X, y_good, "strict_model")

        # Pequeña degradación que supera umbral del 1%
        y_slightly_worse = pd.Series(3 * X["x"].values + np.random.randn(n) * 2.0)
        worse_model = LinearRegression()
        worse_model.fit(X, y_slightly_worse)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor_strict.evaluate_and_monitor(worse_model, X, y_slightly_worse, "strict_model")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1


# ---------------------------------------------------------------------------
# get_performance_history
# ---------------------------------------------------------------------------

class TestGetPerformanceHistory:
    def test_returns_empty_list_initially(self, monitor):
        assert monitor.get_performance_history() == []

    def test_returns_all_entries_without_filter(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "model_a")
        monitor.evaluate_and_monitor(model, X, y, "model_b")
        history = monitor.get_performance_history()
        assert len(history) == 2

    def test_filters_by_model_name(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "model_a")
        monitor.evaluate_and_monitor(model, X, y, "model_b")
        monitor.evaluate_and_monitor(model, X, y, "model_a")
        history_a = monitor.get_performance_history("model_a")
        assert len(history_a) == 2
        assert all(e["model_name"] == "model_a" for e in history_a)

    def test_filter_returns_empty_for_unknown_model(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "model_a")
        assert monitor.get_performance_history("nonexistent") == []

    def test_history_entries_contain_timestamp(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        history = monitor.get_performance_history()
        assert "timestamp" in history[0]

    def test_history_entries_contain_metrics(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        entry = monitor.get_performance_history()[0]
        assert "rmse" in entry
        assert "mae" in entry
        assert "r_squared" in entry

    def test_history_preserves_insertion_order(self, monitor, trained_model):
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "first")
        monitor.evaluate_and_monitor(model, X, y, "second")
        history = monitor.get_performance_history()
        assert history[0]["model_name"] == "first"
        assert history[1]["model_name"] == "second"

    def test_returns_copy_not_reference(self, monitor, trained_model):
        """Modificar la lista retornada no debe afectar el historial interno."""
        model, X, y = trained_model
        monitor.evaluate_and_monitor(model, X, y, "lr")
        history = monitor.get_performance_history()
        history.clear()
        assert len(monitor.history) == 1
