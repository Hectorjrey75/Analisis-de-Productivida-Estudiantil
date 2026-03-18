"""
Tests unitarios para src/recommendations/generator.py.
Cubre: RecommendationEngine.analyze_student_profile,
       generate_recommendations, estimate_impact.
Requisitos: 5.1, 5.3, 5.4
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.recommendations.generator import RecommendationEngine


# ---------------------------------------------------------------------------
# Configuración de rangos óptimos (igual que config/config.yaml)
# ---------------------------------------------------------------------------

OPTIMAL_RANGES = {
    "study_hours_per_day": [6.0, 9.0],
    "sleep_hours": [7.0, 9.0],
    "phone_usage_hours": [0.0, 3.0],
    "social_media_hours": [0.0, 2.0],
    "exercise_minutes": [30, 60],
}

FEATURES = list(OPTIMAL_RANGES.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_importance_df():
    """DataFrame de importancia de features sintético."""
    importances = [0.35, 0.25, 0.20, 0.12, 0.08]
    return pd.DataFrame(
        {"feature": FEATURES, "importance": importances, "rank": range(1, 6)}
    )


@pytest.fixture
def trained_model(feature_importance_df):
    """Modelo lineal simple entrenado con datos sintéticos."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(
        {
            "study_hours_per_day": np.random.uniform(1, 12, n),
            "sleep_hours": np.random.uniform(4, 11, n),
            "phone_usage_hours": np.random.uniform(0, 8, n),
            "social_media_hours": np.random.uniform(0, 6, n),
            "exercise_minutes": np.random.uniform(0, 120, n),
        }
    )
    y = (
        5 * X["study_hours_per_day"]
        + 3 * X["sleep_hours"]
        - 2 * X["phone_usage_hours"]
        - 1.5 * X["social_media_hours"]
        + 0.1 * X["exercise_minutes"]
        + np.random.randn(n) * 2
    )
    model = LinearRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def engine(trained_model, feature_importance_df):
    """RecommendationEngine listo para usar."""
    return RecommendationEngine(
        model=trained_model,
        feature_importance=feature_importance_df,
        optimal_ranges=OPTIMAL_RANGES,
    )


@pytest.fixture
def student_below_optimal():
    """Estudiante con valores por debajo del rango óptimo en varias features."""
    return pd.Series(
        {
            "study_hours_per_day": 3.0,   # óptimo: 6-9
            "sleep_hours": 5.0,           # óptimo: 7-9
            "phone_usage_hours": 1.5,     # óptimo: 0-3 (dentro)
            "social_media_hours": 1.0,    # óptimo: 0-2 (dentro)
            "exercise_minutes": 10.0,     # óptimo: 30-60
        }
    )


@pytest.fixture
def student_above_optimal():
    """Estudiante con valores por encima del rango óptimo."""
    return pd.Series(
        {
            "study_hours_per_day": 7.0,   # dentro
            "sleep_hours": 8.0,           # dentro
            "phone_usage_hours": 6.0,     # óptimo: 0-3 (encima)
            "social_media_hours": 5.0,    # óptimo: 0-2 (encima)
            "exercise_minutes": 45.0,     # dentro
        }
    )


@pytest.fixture
def student_optimal():
    """Estudiante con todos los valores dentro del rango óptimo."""
    return pd.Series(
        {
            "study_hours_per_day": 7.5,
            "sleep_hours": 8.0,
            "phone_usage_hours": 1.5,
            "social_media_hours": 1.0,
            "exercise_minutes": 45.0,
        }
    )


# ---------------------------------------------------------------------------
# Tests para RecommendationEngine.__init__
# ---------------------------------------------------------------------------

class TestRecommendationEngineInit:
    def test_stores_model(self, engine, trained_model):
        assert engine.model is trained_model

    def test_stores_optimal_ranges(self, engine):
        assert engine.optimal_ranges == OPTIMAL_RANGES

    def test_importance_index_populated(self, engine):
        assert len(engine._importance_index) == len(FEATURES)

    def test_importance_index_values(self, engine, feature_importance_df):
        for _, row in feature_importance_df.iterrows():
            assert engine._importance_index[row["feature"]] == pytest.approx(row["importance"])


# ---------------------------------------------------------------------------
# Tests para analyze_student_profile()  — Requisito 5.3
# ---------------------------------------------------------------------------

class TestAnalyzeStudentProfile:
    def test_returns_dict(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        assert isinstance(result, dict)

    def test_has_required_keys(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        assert set(result.keys()) == {"gaps", "below_optimal", "above_optimal"}

    def test_identifies_below_optimal_features(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        assert "study_hours_per_day" in result["below_optimal"]
        assert "sleep_hours" in result["below_optimal"]
        assert "exercise_minutes" in result["below_optimal"]

    def test_identifies_above_optimal_features(self, engine, student_above_optimal):
        result = engine.analyze_student_profile(student_above_optimal)
        assert "phone_usage_hours" in result["above_optimal"]
        assert "social_media_hours" in result["above_optimal"]

    def test_optimal_student_has_no_gaps(self, engine, student_optimal):
        result = engine.analyze_student_profile(student_optimal)
        assert result["below_optimal"] == []
        assert result["above_optimal"] == []
        assert result["gaps"] == {}

    def test_gap_calculation_below(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        # study_hours_per_day: óptimo_min=6.0, actual=3.0 → gap=3.0
        assert result["gaps"]["study_hours_per_day"] == pytest.approx(3.0)

    def test_gap_calculation_above(self, engine, student_above_optimal):
        result = engine.analyze_student_profile(student_above_optimal)
        # phone_usage_hours: óptimo_max=3.0, actual=6.0 → gap=3.0
        assert result["gaps"]["phone_usage_hours"] == pytest.approx(3.0)

    def test_within_optimal_not_in_gaps(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        # phone_usage_hours=1.5 está dentro de [0, 3]
        assert "phone_usage_hours" not in result["gaps"]

    def test_ignores_unknown_features(self, engine):
        """Features no presentes en optimal_ranges deben ignorarse."""
        student = pd.Series(
            {
                "study_hours_per_day": 3.0,
                "unknown_feature": 999.0,
            }
        )
        result = engine.analyze_student_profile(student)
        assert "unknown_feature" not in result["gaps"]

    def test_gaps_are_positive(self, engine, student_below_optimal):
        result = engine.analyze_student_profile(student_below_optimal)
        for gap in result["gaps"].values():
            assert gap > 0


# ---------------------------------------------------------------------------
# Tests para generate_recommendations()  — Requisitos 5.1, 5.2, 5.5
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    def test_returns_list(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        assert isinstance(result, list)

    def test_generates_at_least_3_recommendations(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        assert len(result) >= 3

    def test_each_recommendation_has_required_keys(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        required_keys = {
            "feature", "description", "current_value",
            "target_value", "estimated_impact", "priority",
        }
        for rec in result:
            assert required_keys.issubset(set(rec.keys()))

    def test_priority_values_are_valid(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        for rec in result:
            assert rec["priority"] in {1, 2, 3}

    def test_description_is_non_empty_string(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        for rec in result:
            assert isinstance(rec["description"], str)
            assert len(rec["description"]) > 0

    def test_current_value_matches_student_data(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        for rec in result:
            feature = rec["feature"]
            assert rec["current_value"] == pytest.approx(float(student_below_optimal[feature]))

    def test_target_value_within_optimal_range(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        for rec in result:
            feature = rec["feature"]
            opt_min, opt_max = OPTIMAL_RANGES[feature]
            assert opt_min <= rec["target_value"] <= opt_max

    def test_n_recommendations_parameter(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal, n_recommendations=2)
        assert len(result) <= 2

    def test_empty_result_for_optimal_student(self, engine, student_optimal):
        result = engine.generate_recommendations(student_optimal)
        assert result == []

    def test_estimated_impact_is_float(self, engine, student_below_optimal):
        result = engine.generate_recommendations(student_below_optimal)
        for rec in result:
            assert isinstance(rec["estimated_impact"], float)

    def test_recommendations_for_above_optimal(self, engine, student_above_optimal):
        result = engine.generate_recommendations(student_above_optimal)
        assert len(result) >= 1
        features = [r["feature"] for r in result]
        # phone_usage y social_media están por encima del óptimo
        assert any(f in features for f in ["phone_usage_hours", "social_media_hours"])

    def test_first_recommendation_has_highest_priority(self, engine, student_below_optimal):
        """La primera recomendación debe tener prioridad 1 (más alta)."""
        result = engine.generate_recommendations(student_below_optimal)
        assert result[0]["priority"] == 1


# ---------------------------------------------------------------------------
# Tests para estimate_impact()  — Requisito 5.4
# ---------------------------------------------------------------------------

class TestEstimateImpact:
    def test_returns_float(self, engine, student_below_optimal):
        rec = {
            "feature": "study_hours_per_day",
            "target_value": 7.0,
        }
        result = engine.estimate_impact(student_below_optimal, rec)
        assert isinstance(result, float)

    def test_positive_impact_for_beneficial_change(self, engine, student_below_optimal):
        """Aumentar horas de estudio debe mejorar el score (impacto positivo)."""
        rec = {
            "feature": "study_hours_per_day",
            "target_value": 7.0,
        }
        result = engine.estimate_impact(student_below_optimal, rec)
        # El modelo tiene coeficiente positivo para study_hours_per_day
        assert result > 0

    def test_positive_impact_reducing_phone_usage(self, engine, student_above_optimal):
        """Reducir uso del teléfono debe mejorar el score."""
        rec = {
            "feature": "phone_usage_hours",
            "target_value": 2.0,
        }
        result = engine.estimate_impact(student_above_optimal, rec)
        assert result > 0

    def test_zero_impact_when_no_change(self, engine, student_below_optimal):
        """Si target_value == current_value, el impacto debe ser ~0."""
        current_val = float(student_below_optimal["study_hours_per_day"])
        rec = {
            "feature": "study_hours_per_day",
            "target_value": current_val,
        }
        result = engine.estimate_impact(student_below_optimal, rec)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_does_not_modify_original_student_data(self, engine, student_below_optimal):
        """estimate_impact no debe modificar el Series original."""
        original_value = float(student_below_optimal["study_hours_per_day"])
        rec = {
            "feature": "study_hours_per_day",
            "target_value": 8.0,
        }
        engine.estimate_impact(student_below_optimal, rec)
        assert float(student_below_optimal["study_hours_per_day"]) == pytest.approx(original_value)

    def test_impact_is_finite(self, engine, student_below_optimal):
        rec = {
            "feature": "sleep_hours",
            "target_value": 8.0,
        }
        result = engine.estimate_impact(student_below_optimal, rec)
        assert np.isfinite(result)

    def test_impact_with_random_forest_model(self, feature_importance_df, student_below_optimal):
        """Verificar que estimate_impact funciona con RandomForest."""
        np.random.seed(0)
        n = 200
        X = pd.DataFrame(
            {
                "study_hours_per_day": np.random.uniform(1, 12, n),
                "sleep_hours": np.random.uniform(4, 11, n),
                "phone_usage_hours": np.random.uniform(0, 8, n),
                "social_media_hours": np.random.uniform(0, 6, n),
                "exercise_minutes": np.random.uniform(0, 120, n),
            }
        )
        y = 5 * X["study_hours_per_day"] + 3 * X["sleep_hours"] + np.random.randn(n)
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)

        eng = RecommendationEngine(
            model=rf,
            feature_importance=feature_importance_df,
            optimal_ranges=OPTIMAL_RANGES,
        )
        rec = {"feature": "study_hours_per_day", "target_value": 7.0}
        result = eng.estimate_impact(student_below_optimal, rec)
        assert isinstance(result, float)
        assert np.isfinite(result)
