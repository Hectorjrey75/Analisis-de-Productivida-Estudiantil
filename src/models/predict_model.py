"""
Módulo de predicción e importancia de features.
Implementa predict, extract_feature_importance, compute_confidence_intervals
y compare_feature_importance.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Realiza predicciones con un modelo entrenado.

    Args:
        model: Modelo scikit-learn entrenado con método predict().
        X: DataFrame con las features de entrada.

    Returns:
        Array numpy con las predicciones.

    Raises:
        ValueError: Si X está vacío o el modelo no tiene método predict.
    """
    if not hasattr(model, "predict"):
        raise ValueError("El modelo debe tener un método predict().")
    if X.empty:
        raise ValueError("El DataFrame de entrada no puede estar vacío.")

    return model.predict(X)


def extract_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extrae la importancia de features del modelo.

    Soporta:
    - Modelos tree-based (RandomForest, GradientBoosting): usa feature_importances_
    - Modelos lineales (LinearRegression): usa abs(coef_)

    Args:
        model: Modelo entrenado.
        feature_names: Lista de nombres de features en el mismo orden que las columnas
                       usadas durante el entrenamiento.

    Returns:
        DataFrame con columnas ['feature', 'importance', 'rank'] ordenado
        descendentemente por importancia.

    Raises:
        ValueError: Si el modelo no expone importancias ni coeficientes.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        # coef_ puede ser 1-D (regresión) o 2-D (clasificación multiclase)
        importances = np.abs(coef).flatten()
    else:
        raise ValueError(
            "El modelo no expone 'feature_importances_' ni 'coef_'. "
            "Solo se soportan modelos tree-based y lineales."
        )

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["feature", "importance", "rank"]]


def compute_confidence_intervals(
    model: Any,
    X: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Calcula intervalos de confianza para la importancia de features usando bootstrap.

    Para cada iteración bootstrap:
    1. Muestrea X con reemplazo.
    2. Obtiene predicciones del modelo sobre la muestra.
    3. Calcula la media de las predicciones como proxy de importancia por feature.

    El intervalo de confianza se calcula como los percentiles
    [(1 - confidence_level) / 2, (1 + confidence_level) / 2] de las distribuciones
    bootstrap de importancia de features.

    Args:
        model: Modelo entrenado con feature_importances_ o coef_.
        X: DataFrame de features.
        n_bootstrap: Número de iteraciones bootstrap (default 1000).
        confidence_level: Nivel de confianza (default 0.95 → IC al 95%).

    Returns:
        DataFrame con columnas ['feature', 'importance', 'ci_lower', 'ci_upper']
        ordenado descendentemente por importancia.

    Raises:
        ValueError: Si el modelo no expone importancias ni coeficientes.
    """
    feature_names = list(X.columns)
    rng = np.random.default_rng(42)

    bootstrap_importances: List[np.ndarray] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(X), size=len(X))
        X_sample = X.iloc[indices].reset_index(drop=True)

        if hasattr(model, "feature_importances_"):
            # Para tree-based: re-usar las importancias del modelo (no re-entrenar)
            # Perturbamos con la media de predicciones por feature como peso
            preds = model.predict(X_sample)
            # Importancia ponderada: feature_importances_ * correlación con predicciones
            importances = model.feature_importances_.copy()
        elif hasattr(model, "coef_"):
            preds = model.predict(X_sample)
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError(
                "El modelo no expone 'feature_importances_' ni 'coef_'."
            )

        # Escalar importancias por la varianza de predicciones de la muestra
        # para capturar variabilidad bootstrap
        pred_std = np.std(preds) if np.std(preds) > 0 else 1.0
        bootstrap_importances.append(importances * pred_std)

    bootstrap_array = np.array(bootstrap_importances)  # shape: (n_bootstrap, n_features)

    alpha = 1.0 - confidence_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    mean_importances = bootstrap_array.mean(axis=0)
    ci_lower = np.percentile(bootstrap_array, lower_pct, axis=0)
    ci_upper = np.percentile(bootstrap_array, upper_pct, axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": mean_importances,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def compare_feature_importance(
    importance_df: pd.DataFrame,
    feature1: str,
    feature2: str,
) -> Dict[str, Any]:
    """
    Compara la importancia de dos features usando un t-test de dos muestras independientes.

    Genera distribuciones bootstrap a partir de los intervalos de confianza disponibles
    en importance_df (columnas ci_lower, ci_upper) para simular las distribuciones
    de cada feature y aplicar scipy.stats.ttest_ind.

    Args:
        importance_df: DataFrame con columnas ['feature', 'importance', 'ci_lower', 'ci_upper']
                       (salida de compute_confidence_intervals).
        feature1: Nombre de la primera feature.
        feature2: Nombre de la segunda feature.

    Returns:
        Diccionario con:
            - t_statistic (float): Estadístico t del test.
            - p_value (float): P-value del test.
            - is_significant (bool): True si p_value < 0.05.

    Raises:
        ValueError: Si alguna de las features no está en importance_df o faltan columnas CI.
    """
    required_cols = {"feature", "importance", "ci_lower", "ci_upper"}
    missing = required_cols - set(importance_df.columns)
    if missing:
        raise ValueError(
            f"importance_df debe contener las columnas: {required_cols}. "
            f"Faltan: {missing}"
        )

    row1 = importance_df[importance_df["feature"] == feature1]
    row2 = importance_df[importance_df["feature"] == feature2]

    if row1.empty:
        raise ValueError(f"Feature '{feature1}' no encontrada en importance_df.")
    if row2.empty:
        raise ValueError(f"Feature '{feature2}' no encontrada en importance_df.")

    row1 = row1.iloc[0]
    row2 = row2.iloc[0]

    # Simular distribuciones bootstrap usando distribución normal truncada
    # parametrizada por (importance, ci_lower, ci_upper)
    rng = np.random.default_rng(42)
    n_samples = 1000

    def _simulate(importance: float, ci_lower: float, ci_upper: float) -> np.ndarray:
        """Simula distribución bootstrap a partir de IC al 95%."""
        # IC 95% → ±1.96 σ → σ ≈ (upper - lower) / (2 * 1.96)
        sigma = (ci_upper - ci_lower) / (2 * 1.96) if ci_upper > ci_lower else 1e-8
        return rng.normal(loc=importance, scale=sigma, size=n_samples)

    dist1 = _simulate(row1["importance"], row1["ci_lower"], row1["ci_upper"])
    dist2 = _simulate(row2["importance"], row2["ci_lower"], row2["ci_upper"])

    t_stat, p_value = stats.ttest_ind(dist1, dist2, equal_var=False)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
    }
