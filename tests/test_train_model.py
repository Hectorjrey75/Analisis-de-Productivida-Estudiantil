"""
Tests unitarios para src/models/train_model.py.
Cubre ModelTrainer: constructor, split_data, train_model, cross_validate, evaluate_model.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.train_model import ModelTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config():
    """Configuración mínima válida para ModelTrainer."""
    return {
        "models": {
            "types": ["linear_regression", "random_forest", "gradient_boosting"],
            "regression": {
                "linear_regression": {},
                "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
                "gradient_boosting": {
                    "n_estimators": 10,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
            },
            "production_threshold": {"r_squared": 0.7},
        },
        "training": {
            "test_size": 0.15,
            "validation_size": 0.15,
            "random_state": 42,
            "cv_folds": 5,
        },
    }


@pytest.fixture
def trainer(base_config):
    return ModelTrainer(base_config)


@pytest.fixture
def synthetic_data():
    """Dataset sintético con relación lineal fuerte para facilitar pruebas."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "study_hours": np.random.uniform(1, 10, n),
            "sleep_hours": np.random.uniform(4, 10, n),
            "phone_usage": np.random.uniform(0, 8, n),
        }
    )
    # Relación lineal con ruido pequeño → R² alto
    y = pd.Series(
        3 * X["study_hours"] + 2 * X["sleep_hours"] - X["phone_usage"] + np.random.randn(n) * 0.5,
        name="productivity_score",
    )
    return X, y


# ---------------------------------------------------------------------------
# 7.1 Constructor
# ---------------------------------------------------------------------------

class TestModelTrainerConstructor:
    def test_creates_instance(self, base_config):
        trainer = ModelTrainer(base_config)
        assert isinstance(trainer, ModelTrainer)

    def test_models_dict_initialized_empty(self, trainer):
        assert isinstance(trainer.models, dict)
        assert len(trainer.models) == 0

    def test_results_dict_initialized_empty(self, trainer):
        assert isinstance(trainer.results, dict)
        assert len(trainer.results) == 0

    def test_loads_training_config(self, trainer):
        assert trainer.test_size == 0.15
        assert trainer.validation_size == 0.15
        assert trainer.random_state == 42
        assert trainer.cv_folds == 5

    def test_loads_production_threshold(self, trainer):
        assert trainer.production_threshold == 0.7

    def test_loads_model_types(self, trainer):
        assert "linear_regression" in trainer.model_types
        assert "random_forest" in trainer.model_types
        assert "gradient_boosting" in trainer.model_types

    def test_loads_hyperparams_random_forest(self, trainer):
        rf_params = trainer.hyperparams["random_forest"]
        assert rf_params["n_estimators"] == 10
        assert rf_params["random_state"] == 42

    def test_defaults_when_config_empty(self):
        trainer = ModelTrainer({})
        assert trainer.test_size == 0.15
        assert trainer.validation_size == 0.15
        assert trainer.random_state == 42
        assert trainer.cv_folds == 5
        assert trainer.production_threshold == 0.7


# ---------------------------------------------------------------------------
# 7.2 split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_returns_six_elements(self, trainer, synthetic_data):
        X, y = synthetic_data
        result = trainer.split_data(X, y)
        assert len(result) == 6

    def test_total_samples_preserved(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

    def test_test_proportion_approx_15_percent(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, _, _, _ = trainer.split_data(X, y)
        test_ratio = len(X_test) / len(X)
        assert abs(test_ratio - 0.15) < 0.05

    def test_val_proportion_approx_15_percent(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, _, _, _ = trainer.split_data(X, y)
        val_ratio = len(X_val) / len(X)
        assert abs(val_ratio - 0.15) < 0.05

    def test_train_proportion_approx_70_percent(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, _, _, _ = trainer.split_data(X, y)
        train_ratio = len(X_train) / len(X)
        assert abs(train_ratio - 0.70) < 0.05

    def test_no_overlap_between_splits(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, _, _, _ = trainer.split_data(X, y)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_reproducible_with_same_random_state(self, trainer, synthetic_data):
        X, y = synthetic_data
        split1 = trainer.split_data(X, y)
        split2 = trainer.split_data(X, y)
        pd.testing.assert_frame_equal(split1[0], split2[0])  # X_train

    def test_x_and_y_splits_aligned(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
        assert list(X_train.index) == list(y_train.index)
        assert list(X_val.index) == list(y_val.index)
        assert list(X_test.index) == list(y_test.index)

    def test_custom_sizes_respected(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, X_val, X_test, _, _, _ = trainer.split_data(X, y, test_size=0.2, val_size=0.2)
        test_ratio = len(X_test) / len(X)
        val_ratio = len(X_val) / len(X)
        assert abs(test_ratio - 0.20) < 0.05
        assert abs(val_ratio - 0.20) < 0.05


# ---------------------------------------------------------------------------
# 7.3 train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    @pytest.mark.parametrize("model_type", ["linear_regression", "random_forest", "gradient_boosting"])
    def test_returns_fitted_model(self, trainer, synthetic_data, model_type):
        X, y = synthetic_data
        X_train, _, _, y_train, _, _ = trainer.split_data(X, y)
        model = trainer.train_model(model_type, X_train, y_train)
        assert model is not None

    @pytest.mark.parametrize("model_type", ["linear_regression", "random_forest", "gradient_boosting"])
    def test_model_stored_in_dict(self, trainer, synthetic_data, model_type):
        X, y = synthetic_data
        X_train, _, _, y_train, _, _ = trainer.split_data(X, y)
        trainer.train_model(model_type, X_train, y_train)
        assert model_type in trainer.models

    @pytest.mark.parametrize("model_type", ["linear_regression", "random_forest", "gradient_boosting"])
    def test_model_can_predict(self, trainer, synthetic_data, model_type):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, _ = trainer.split_data(X, y)
        model = trainer.train_model(model_type, X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_raises_for_unknown_model_type(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, _, y_train, _, _ = trainer.split_data(X, y)
        with pytest.raises(ValueError, match="no reconocido"):
            trainer.train_model("unknown_model", X_train, y_train)

    def test_custom_hyperparams_applied(self, trainer, synthetic_data):
        """Hiperparámetros personalizados deben sobreescribir los de config."""
        X, y = synthetic_data
        X_train, _, _, y_train, _, _ = trainer.split_data(X, y)
        model = trainer.train_model(
            "random_forest", X_train, y_train, hyperparams={"n_estimators": 5}
        )
        assert model.n_estimators == 5

    def test_linear_regression_has_coef(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, _, y_train, _, _ = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        assert hasattr(model, "coef_")
        assert len(model.coef_) == X_train.shape[1]


# ---------------------------------------------------------------------------
# 7.4 cross_validate
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_returns_dict_with_required_keys(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y)
        assert "cv_scores" in result
        assert "cv_mean" in result
        assert "cv_std" in result

    def test_cv_scores_length_equals_folds(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y, cv_folds=5)
        assert len(result["cv_scores"]) == 5

    def test_cv_mean_is_average_of_scores(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y)
        assert result["cv_mean"] == pytest.approx(np.mean(result["cv_scores"]), abs=1e-6)

    def test_cv_std_is_std_of_scores(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y)
        assert result["cv_std"] == pytest.approx(np.std(result["cv_scores"]), abs=1e-6)

    def test_cv_scores_are_floats(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y)
        for score in result["cv_scores"]:
            assert isinstance(score, float)

    def test_custom_cv_folds(self, trainer, synthetic_data):
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y, cv_folds=3)
        assert len(result["cv_scores"]) == 3

    def test_high_r2_for_linear_data(self, trainer, synthetic_data):
        """Datos con relación lineal fuerte deben dar R² alto en CV."""
        X, y = synthetic_data
        model = LinearRegression()
        result = trainer.cross_validate(model, X, y)
        assert result["cv_mean"] > 0.9


# ---------------------------------------------------------------------------
# 7.5 evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_dict_with_required_keys(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert "rmse" in result
        assert "mae" in result
        assert "r_squared" in result
        assert "is_production_candidate" in result

    def test_rmse_is_non_negative(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert result["rmse"] >= 0.0

    def test_mae_is_non_negative(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert result["mae"] >= 0.0

    def test_r_squared_range(self, trainer, synthetic_data):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert result["r_squared"] <= 1.0

    def test_is_production_candidate_true_when_r2_above_threshold(self, trainer, synthetic_data):
        """Datos lineales fuertes → R² > 0.7 → candidato para producción."""
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        # Con datos lineales fuertes, R² debe ser > 0.7
        if result["r_squared"] > 0.7:
            assert result["is_production_candidate"] is True
        else:
            assert result["is_production_candidate"] is False

    def test_is_production_candidate_false_when_r2_below_threshold(self, trainer):
        """Modelo con predicciones aleatorias → R² bajo → no candidato."""
        np.random.seed(0)
        n = 50
        X_test = pd.DataFrame({"a": np.random.randn(n)})
        y_test = pd.Series(np.random.randn(n))

        # Modelo entrenado con datos no relacionados
        X_train = pd.DataFrame({"a": np.random.randn(100)})
        y_train = pd.Series(np.random.randn(100))
        model = trainer.train_model("linear_regression", X_train, y_train)

        result = trainer.evaluate_model(model, X_test, y_test)
        assert result["is_production_candidate"] is False

    def test_rmse_mae_consistent(self, trainer, synthetic_data):
        """RMSE >= MAE siempre (por desigualdad de Jensen)."""
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model("linear_regression", X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert result["rmse"] >= result["mae"]

    @pytest.mark.parametrize("model_type", ["linear_regression", "random_forest", "gradient_boosting"])
    def test_all_model_types_evaluate(self, trainer, synthetic_data, model_type):
        X, y = synthetic_data
        X_train, _, X_test, y_train, _, y_test = trainer.split_data(X, y)
        model = trainer.train_model(model_type, X_train, y_train)
        result = trainer.evaluate_model(model, X_test, y_test)
        assert isinstance(result["rmse"], float)
        assert isinstance(result["mae"], float)
        assert isinstance(result["r_squared"], float)
        assert isinstance(result["is_production_candidate"], bool)
