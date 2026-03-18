"""
Módulo de ingeniería de features para el sistema de análisis de productividad estudiantil.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


def normalize_features(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Normaliza features numéricas del DataFrame.

    Args:
        df: DataFrame con features a normalizar.
        method: 'standard' para z-score (StandardScaler) o 'minmax' para Min-Max scaling.

    Returns:
        Tupla (DataFrame normalizado, scaler ajustado).

    Raises:
        ValueError: Si el método especificado no es válido.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Método de normalización no válido: '{method}'. Use 'standard' o 'minmax'.")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    normalized_df = df.copy()
    if numeric_cols:
        normalized_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return normalized_df, scaler


def encode_categorical(df: pd.DataFrame, method: str = 'onehot') -> Tuple[pd.DataFrame, Any]:
    """
    Codifica variables categóricas del DataFrame.

    Args:
        df: DataFrame con variables categóricas a codificar.
        method: 'onehot' para One-Hot Encoding (OneHotEncoder, sparse_output=False)
                o 'label' para Label Encoding (un LabelEncoder por columna).

    Returns:
        Tupla (DataFrame codificado, encoder ajustado).
        - Para 'onehot': encoder es un OneHotEncoder ajustado.
        - Para 'label': encoder es un dict {columna: LabelEncoder ajustado}.

    Raises:
        ValueError: Si el método especificado no es válido.
    """
    if method not in ('onehot', 'label'):
        raise ValueError(f"Método de codificación no válido: '{method}'. Use 'onehot' o 'label'.")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoded_df = df.copy()

    if not cat_cols:
        # No hay columnas categóricas; retornar encoder vacío apropiado
        if method == 'onehot':
            encoder: Any = OneHotEncoder(sparse_output=False)
        else:
            encoder = {}
        return encoded_df, encoder

    if method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False)
        encoded_array = encoder.fit_transform(df[cat_cols])
        new_col_names = encoder.get_feature_names_out(cat_cols).tolist()
        encoded_df = df.drop(columns=cat_cols).reset_index(drop=True)
        ohe_df = pd.DataFrame(encoded_array, columns=new_col_names)
        encoded_df = pd.concat([encoded_df, ohe_df], axis=1)
    else:  # label
        encoders: Dict[str, LabelEncoder] = {}
        for col in cat_cols:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col])
            encoders[col] = le
        encoder = encoders

    return encoded_df, encoder


def handle_outliers(df: pd.DataFrame, strategy: str = 'clip', threshold: float = 3.0) -> pd.DataFrame:
    """
    Maneja outliers en columnas numéricas del DataFrame.

    Args:
        df: DataFrame con datos.
        strategy: Estrategia para manejar outliers:
            - 'clip': limita valores al rango [mean - threshold*std, mean + threshold*std].
            - 'remove': elimina filas donde cualquier columna numérica supera el umbral.
            - 'winsorize': recorta valores a percentiles equivalentes al umbral de std.
        threshold: Número de desviaciones estándar que define el umbral de outlier.

    Returns:
        DataFrame con outliers manejados.

    Raises:
        ValueError: Si la estrategia especificada no es válida.
    """
    if strategy not in ('clip', 'remove', 'winsorize'):
        raise ValueError(
            f"Estrategia no válida: '{strategy}'. Use 'clip', 'remove' o 'winsorize'."
        )

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    result_df = df.copy()

    if not numeric_cols:
        return result_df

    if strategy == 'clip':
        for col in numeric_cols:
            mean = result_df[col].mean()
            std = result_df[col].std()
            lower = mean - threshold * std
            upper = mean + threshold * std
            result_df[col] = result_df[col].clip(lower=lower, upper=upper)

    elif strategy == 'remove':
        mask = pd.Series([True] * len(result_df), index=result_df.index)
        for col in numeric_cols:
            mean = result_df[col].mean()
            std = result_df[col].std()
            lower = mean - threshold * std
            upper = mean + threshold * std
            mask &= result_df[col].between(lower, upper)
        result_df = result_df[mask].reset_index(drop=True)

    elif strategy == 'winsorize':
        import math
        for col in numeric_cols:
            series = result_df[col]
            lower_pct = float(np.percentile(series, 100 * _norm_cdf(-threshold)))
            upper_pct = float(np.percentile(series, 100 * _norm_cdf(threshold)))
            result_df[col] = series.clip(lower=lower_pct, upper=upper_pct)

    return result_df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas automáticamente a partir de columnas existentes.

    Features creadas (solo si las columnas fuente existen):
        - total_screen_time = phone_usage_hours + social_media_hours + youtube_hours + gaming_hours
        - study_sleep_ratio = study_hours_per_day / sleep_hours
        - digital_distraction_index = (social_media_hours + gaming_hours) / study_hours_per_day
        - screen_to_study_ratio = total_screen_time / study_hours_per_day

    Args:
        df: DataFrame con columnas originales.

    Returns:
        DataFrame con columnas originales más las features derivadas calculadas.
    """
    result = df.copy()
    cols = set(df.columns)

    screen_cols = ['phone_usage_hours', 'social_media_hours', 'youtube_hours', 'gaming_hours']
    if all(c in cols for c in screen_cols):
        result['total_screen_time'] = (
            df['phone_usage_hours'] + df['social_media_hours']
            + df['youtube_hours'] + df['gaming_hours']
        )

    if 'study_hours_per_day' in cols and 'sleep_hours' in cols:
        result['study_sleep_ratio'] = (
            df['study_hours_per_day'] / df['sleep_hours']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if 'social_media_hours' in cols and 'gaming_hours' in cols and 'study_hours_per_day' in cols:
        result['digital_distraction_index'] = (
            (df['social_media_hours'] + df['gaming_hours']) / df['study_hours_per_day']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    if 'total_screen_time' in result.columns and 'study_hours_per_day' in cols:
        result['screen_to_study_ratio'] = (
            result['total_screen_time'] / df['study_hours_per_day']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return result


def _norm_cdf(x: float) -> float:
    """Aproximación de la CDF normal estándar usando math.erf."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
