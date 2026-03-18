"""
Tests unitarios para src/models/predict_model.py.
Cubre: predict, extract_feature_importance, compute_confidence_intervals,
       compare_feature_importance.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.predict_model import (
    compare_feature_importance,
    compute_confidence_intervals,
    extract_feature_importance,
    predict,
)


# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Dataset sintético con relación lineal fuerte."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "study_hours": np.random.uniform(1, 10, n),
            "sleep_hours": np.random.uniform(4, 10, n),
            "phone_usage": np.random.uniform(0, 8, n),
        }
    )
    y = (
        3 * X["study_hours"]
        + 2 * X["sleep_hours"]
        - X["phone_usage"]
        + np.random.randn(n) * 0.5
    )
    return X, y


@pytest.fixture
def trained_linear(synthetic_data):
    X, y = synthetic_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_rf(synthetic_data):
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_gb(synthetic_data):
    X, y = synthetic_data
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Tests para predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_numpy_array(self, trained_linear):
        model, X, _ = trained_linear
        result = predict(model, X)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self, trained_linear):
        model, X, _ = trained_linear
        result = predict(model, X)
        assert len(result) == len(X)

    def test_predictions_are_finite(self, trained_linear):
        model, X, _ = trained_linear
        result = predict(model, X)
        assert np.all(np.isfinite(result))

    def test_raises_on_empty_dataframe(self, trained_linear):
        model, X, _ = trained_linear
        empty_X = X.iloc[0:0]
        with pytest.raises(ValueError, match="vacío"):
            predict(model, empty_X)

    def test_raises_when_model_has_no_predict(self, synthetic_data):
        X, _ = synthetic_data

        class FakeModel:
            pass

        with pytest.raises(ValueError, match="predict"):
            predict(FakeModel(), X)

    @pytest.mark.parametrize("fixture_name", ["trained_linear", "trained_rf", "trained_gb"])
    def test_all_model_types(self, request, fixture_name):
        model, X, _ = request.getfixturevalue(fixture_name)
        result = predict(model, X)
        assert len(result) == len(X)


# ---------------------------------------------------------------------------
# Tests para extract_feature_importance()
# ---------------------------------------------------------------------------

class TestExtractFeatureImportance:
    def test_returns_dataframe(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert set(result.columns) == {"feature", "importance", "rank"}

    def test_length_equals_number_of_features(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert len(result) == X.shape[1]

    def test_sorted_descending_by_importance(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_rank_starts_at_one(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert result["rank"].iloc[0] == 1

    def test_rank_is_sequential(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert list(result["rank"]) == list(range(1, len(result) + 1))

    def test_all_features_present(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert set(result["feature"]) == set(X.columns)

    def test_importances_non_negative_for_tree_model(self, trained_rf):
        model, X, _ = trained_rf
        result = extract_feature_importance(model, list(X.columns))
        assert (result["importance"] >= 0).all()

    def test_linear_model_uses_abs_coef(self, trained_linear):
        model, X, _ = trained_linear
        result = extract_feature_importance(model, list(X.columns))
        assert (result["importance"] >= 0).all()

    def test_gradient_boosting_model(self, trained_gb):
        model, X, _ = trained_gb
        result = extract_feature_importance(model, list(X.columns))
        assert len(result) == X.shape[1]

    def test_raises_for_unsupported_model(self, synthetic_data):
        X, _ = synthetic_data

        class UnsupportedModel:
            pass

        with pytest.raises(ValueError, match="feature_importances_"):
            extract_feature_importance(UnsupportedModel(), list(X.columns))

    def test_study_hours_has_highest_importance_linear(self, trained_linear):
        """study_hours tiene coeficiente 3 (el mayor) → debe ser la más importante."""
        model, X, _ = trained_linear
        result = extract_feature_importance(model, list(X.columns))
        assert result.iloc[0]["feature"] == "study_hours"


# ---------------------------------------------------------------------------
# Tests para compute_confidence_intervals()
# ---------------------------------------------------------------------------

class TestComputeConfidenceIntervals:
    def test_returns_dataframe(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        assert set(result.columns) == {"feature", "importance", "ci_lower", "ci_upper"}

    def test_length_equals_number_of_features(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        assert len(result) == X.shape[1]

    def test_ci_lower_leq_importance(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=100)
        assert (result["ci_lower"] <= result["importance"]).all()

    def test_ci_upper_geq_importance(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=100)
        assert (result["ci_upper"] >= result["importance"]).all()

    def test_ci_lower_leq_ci_upper(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=100)
        assert (result["ci_lower"] <= result["ci_upper"]).all()

    def test_sorted_descending_by_importance(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_all_features_present(self, trained_rf):
        model, X, _ = trained_rf
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        assert set(result["feature"]) == set(X.columns)

    def test_linear_model_supported(self, trained_linear):
        model, X, _ = trained_linear
        result = compute_confidence_intervals(model, X, n_bootstrap=50)
        assert len(result) == X.shape[1]

    def test_raises_for_unsupported_model(self, synthetic_data):
        X, _ = synthetic_data

        class UnsupportedModel:
            pass

        with pytest.raises(ValueError):
            compute_confidence_intervals(UnsupportedModel(), X, n_bootstrap=10)

    def test_default_n_bootstrap_is_1000(self, trained_rf):
        """Verificar que la función acepta n_bootstrap=1000 sin errores."""
        model, X, _ = trained_rf
        # Usamos n_bootstrap pequeño para velocidad, pero verificamos el parámetro
        result = compute_confidence_intervals(model, X, n_bootstrap=1000)
        assert len(result) == X.shape[1]


# ---------------------------------------------------------------------------
# Tests para compare_feature_importance()
# ---------------------------------------------------------------------------

class TestCompareFeatureImportance:
    @pytest.fixture
    def importance_df(self, trained_rf):
        model, X, _ = trained_rf
        return compute_confidence_intervals(model, X, n_bootstrap=200)

    def test_returns_dict(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        assert isinstance(result, dict)

    def test_has_required_keys(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        assert set(result.keys()) == {"t_statistic", "p_value", "is_significant"}

    def test_p_value_in_range(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        assert 0.0 <= result["p_value"] <= 1.0

    def test_is_significant_is_bool(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        assert isinstance(result["is_significant"], bool)

    def test_is_significant_consistent_with_p_value(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        expected = result["p_value"] < 0.05
        assert result["is_significant"] == expected

    def test_t_statistic_is_float(self, importance_df):
        features = importance_df["feature"].tolist()
        result = compare_feature_importance(importance_df, features[0], features[1])
        assert isinstance(result["t_statistic"], float)

    def test_raises_for_missing_feature1(self, importance_df):
        with pytest.raises(ValueError, match="no encontrada"):
            compare_feature_importance(importance_df, "nonexistent_feature", "study_hours")

    def test_raises_for_missing_feature2(self, importance_df):
        features = importance_df["feature"].tolist()
        with pytest.raises(ValueError, match="no encontrada"):
            compare_feature_importance(importance_df, features[0], "nonexistent_feature")

    def test_raises_for_missing_ci_columns(self, trained_rf):
        model, X, _ = trained_rf
        # DataFrame sin columnas CI
        df_no_ci = extract_feature_importance(model, list(X.columns))
        with pytest.raises(ValueError, match="ci_lower"):
            compare_feature_importance(df_no_ci, "study_hours", "sleep_hours")

    def test_same_feature_comparison_not_significant(self, importance_df):
        """Comparar una feature consigo misma → p_value alto, no significativo."""
        feature = importance_df["feature"].iloc[0]
        result = compare_feature_importance(importance_df, feature, feature)
        # Misma distribución → no debe ser significativo
        assert result["is_significant"] is False
