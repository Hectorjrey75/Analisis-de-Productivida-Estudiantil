"""Módulo de análisis de correlación estadística."""

from typing import List, Tuple

import pandas as pd


VALID_METHODS = ('pearson', 'spearman', 'kendall')


def compute_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula la matriz de correlación para columnas numéricas.

    Args:
        df: DataFrame con los datos.
        method: Método de correlación: 'pearson', 'spearman' o 'kendall'.

    Returns:
        Matriz de correlación como DataFrame.

    Raises:
        ValueError: Si el método no es válido.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Método '{method}' no válido. Use uno de: {', '.join(VALID_METHODS)}"
        )

    numeric_df = df.select_dtypes(include='number')
    return numeric_df.corr(method=method)


def find_significant_correlations(
    corr_matrix: pd.DataFrame,
    target: str,
    confidence_level: float = 0.95,
    n_samples: int = None,
) -> pd.DataFrame:
    """
    Identifica correlaciones estadísticamente significativas con la variable objetivo.

    Args:
        corr_matrix: Matriz de correlación.
        target: Variable objetivo.
        confidence_level: Nivel de confianza (default 0.95).
        n_samples: Número de muestras para calcular p-values.

    Returns:
        DataFrame con correlaciones significativas y p-values.
    """
    import numpy as np
    from scipy import stats

    if target not in corr_matrix.columns:
        raise ValueError(f"Variable objetivo '{target}' no encontrada en la matriz.")

    alpha = 1 - confidence_level
    target_corr = corr_matrix[target].drop(labels=[target])

    rows = []
    for feature, corr in target_corr.items():
        if np.isnan(corr):
            continue
        if n_samples is not None:
            # t-statistic para correlación de Pearson
            t_stat = corr * np.sqrt(n_samples - 2) / np.sqrt(1 - corr ** 2)
            p_value = 2 * stats.t.sf(abs(t_stat), df=n_samples - 2)
            is_significant = bool(p_value < alpha)
        else:
            p_value = None
            is_significant = None
        rows.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'is_significant': is_significant,
        })

    result = pd.DataFrame(rows, columns=['feature', 'correlation', 'p_value', 'is_significant'])

    # Filtrar solo correlaciones significativas cuando se dispone de p-values
    if n_samples is not None:
        result = result[result['is_significant'] == True]

    # Ordenar por magnitud absoluta de correlación (descendente)
    result = result.reindex(result['correlation'].abs().sort_values(ascending=False).index)
    return result.reset_index(drop=True)


def detect_multicollinearity(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> List[Tuple[str, str, float]]:
    """
    Detecta multicolinealidad entre variables predictoras.

    Args:
        corr_matrix: Matriz de correlación.
        threshold: Umbral de correlación para considerar multicolinealidad.

    Returns:
        Lista de tuplas (var1, var2, correlación).
    """
    pairs = []
    columns = corr_matrix.columns.tolist()
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) >= threshold:
                pairs.append((col1, col2, corr_val))
    return pairs
