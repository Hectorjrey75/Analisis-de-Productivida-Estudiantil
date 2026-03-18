"""
Módulo de entrenamiento de modelos predictivos.
Implementa ModelTrainer para entrenar, validar y evaluar modelos de ML.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


class ModelTrainer:
    """Entrenador de modelos con validación cruzada."""

    def __init__(self, config: Dict):
        """
        Inicializa el entrenador con la configuración de modelos.

        Args:
            config: Diccionario de configuración con secciones 'models' y 'training'.
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}

        # Extraer configuración de entrenamiento
        training_cfg = config.get("training", {})
        self.test_size: float = training_cfg.get("test_size", 0.15)
        self.validation_size: float = training_cfg.get("validation_size", 0.15)
        self.random_state: int = training_cfg.get("random_state", 42)
        self.cv_folds: int = training_cfg.get("cv_folds", 5)

        # Extraer configuración de modelos
        models_cfg = config.get("models", {})
        self.model_types: List[str] = models_cfg.get(
            "types", ["linear_regression", "random_forest", "gradient_boosting"]
        )
        regression_cfg = models_cfg.get("regression", {})
        self.hyperparams: Dict[str, Dict] = {
            "linear_regression": regression_cfg.get("linear_regression", {}),
            "random_forest": regression_cfg.get("random_forest", {}),
            "gradient_boosting": regression_cfg.get("gradient_boosting", {}),
        }
        self.production_threshold: float = (
            models_cfg.get("production_threshold", {}).get("r_squared", 0.7)
        )

    # ------------------------------------------------------------------
    # 7.2 split_data
    # ------------------------------------------------------------------

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Divide datos en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%).

        Args:
            X: Features.
            y: Variable objetivo.
            test_size: Proporción para test (default desde config).
            val_size: Proporción para validación (default desde config).

        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        test_size = test_size if test_size is not None else self.test_size
        val_size = val_size if val_size is not None else self.validation_size

        # Primera división: separar test del resto
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Segunda división: separar validación del conjunto restante
        # val_size relativo al conjunto original → ajustar proporción
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # 7.3 train_model
    # ------------------------------------------------------------------

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparams: Optional[Dict] = None,
    ) -> Any:
        """
        Entrena un modelo del tipo especificado.

        Args:
            model_type: 'linear_regression', 'random_forest' o 'gradient_boosting'.
            X_train: Features de entrenamiento.
            y_train: Target de entrenamiento.
            hyperparams: Hiperparámetros opcionales (sobreescriben los de config).

        Returns:
            Modelo entrenado.

        Raises:
            ValueError: Si model_type no es reconocido.
        """
        # Combinar hiperparámetros de config con los proporcionados
        params = dict(self.hyperparams.get(model_type, {}))
        if hyperparams:
            params.update(hyperparams)

        if model_type == "linear_regression":
            model = LinearRegression(**params)
        elif model_type == "random_forest":
            model = RandomForestRegressor(**params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**params)
        else:
            raise ValueError(
                f"Tipo de modelo no reconocido: '{model_type}'. "
                "Use 'linear_regression', 'random_forest' o 'gradient_boosting'."
            )

        model.fit(X_train, y_train)
        self.models[model_type] = model
        return model

    # ------------------------------------------------------------------
    # 7.4 cross_validate
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Realiza validación cruzada con k folds.

        Args:
            model: Modelo scikit-learn (no necesita estar entrenado).
            X: Features.
            y: Target.
            cv_folds: Número de folds (default desde config).

        Returns:
            Diccionario con 'cv_scores', 'cv_mean' y 'cv_std'.
        """
        cv_folds = cv_folds if cv_folds is not None else self.cv_folds

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

        return {
            "cv_scores": scores.tolist(),
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
        }

    # ------------------------------------------------------------------
    # 7.5 evaluate_model
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo en el conjunto de prueba.

        Args:
            model: Modelo entrenado.
            X_test: Features de prueba.
            y_test: Target de prueba.

        Returns:
            Diccionario con 'rmse', 'mae', 'r_squared' e 'is_production_candidate'.
        """
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r_squared = float(r2_score(y_test, y_pred))
        is_production_candidate = r_squared > self.production_threshold

        return {
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "is_production_candidate": is_production_candidate,
        }
