"""Módulo de ingesta de datos estudiantiles."""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd

from src.models.data_models import ValidationResult

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str, config: Dict) -> pd.DataFrame:
    """
    Carga datos crudos desde archivo CSV.

    Args:
        file_path: Ruta al archivo CSV
        config: Configuración de carga (sección 'data' del config.yaml)

    Returns:
        DataFrame con datos cargados

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato es inválido
    """
    logger.info(f"Cargando datos desde: {file_path}")

    # Verificar existencia del archivo
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    except Exception as exc:
        raise ValueError(f"Formato de archivo inválido en '{file_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"El archivo '{file_path}' no contiene datos.")

    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Validar columnas requeridas según config
    required_columns: List[str] = config.get("data", {}).get("required_columns", [])
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Columnas requeridas faltantes en '{file_path}': {missing}"
            )

    return df


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
    """
    Valida que el DataFrame contenga todas las columnas requeridas.

    Args:
        df: DataFrame a validar
        required_columns: Lista de columnas requeridas

    Returns:
        ValidationResult con estado y mensajes de error
    """
    errors: List[str] = []
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"Columnas requeridas faltantes: {missing}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=[],
        quality_metrics={"total_columns": len(df.columns), "missing_columns": missing},
    )


def compute_data_quality_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula estadísticas de calidad de datos.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con métricas de calidad (valores nulos, duplicados, etc.)
    """
    total_rows = len(df)
    total_columns = len(df.columns)

    # Porcentaje de nulos por columna
    null_percentages: Dict[str, float] = {
        col: round(df[col].isna().sum() / total_rows * 100, 4) if total_rows > 0 else 0.0
        for col in df.columns
    }

    # Filas duplicadas
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage = round(duplicate_rows / total_rows * 100, 4) if total_rows > 0 else 0.0

    # Estadísticas descriptivas por columna numérica
    numeric_cols = df.select_dtypes(include="number").columns
    descriptive_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        descriptive_stats[col] = {
            "mean": float(series.mean()) if len(series) > 0 else float("nan"),
            "std": float(series.std()) if len(series) > 0 else float("nan"),
            "min": float(series.min()) if len(series) > 0 else float("nan"),
            "max": float(series.max()) if len(series) > 0 else float("nan"),
        }

    return {
        "null_percentages": null_percentages,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "descriptive_stats": descriptive_stats,
        "total_rows": total_rows,
        "total_columns": total_columns,
    }


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputa valores faltantes según la estrategia indicada.

    Args:
        df: DataFrame con posibles valores NaN
        strategy: Estrategia de imputación. Opciones:
            - 'mean': rellena columnas numéricas con la media
            - 'median': rellena columnas numéricas con la mediana
            - 'mode': rellena todas las columnas con la moda
            - 'forward_fill': propaga el último valor válido hacia adelante
            - 'drop': elimina filas con cualquier NaN

    Returns:
        DataFrame con valores imputados

    Raises:
        ValueError: Si la estrategia no es válida
    """
    valid_strategies = {'mean', 'median', 'mode', 'forward_fill', 'drop'}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Estrategia de imputación inválida: '{strategy}'. "
            f"Opciones válidas: {sorted(valid_strategies)}"
        )

    result = df.copy()
    numeric_cols = result.select_dtypes(include='number').columns

    if strategy == 'mean':
        for col in numeric_cols:
            result[col] = result[col].fillna(result[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            result[col] = result[col].fillna(result[col].median())
    elif strategy == 'mode':
        for col in result.columns:
            mode_vals = result[col].mode()
            if not mode_vals.empty:
                result[col] = result[col].fillna(mode_vals.iloc[0])
    elif strategy == 'forward_fill':
        result = result.ffill()
    elif strategy == 'drop':
        result = result.dropna()

    logger.info(f"Imputación aplicada con estrategia '{strategy}'")
    return result


def apply_imputation(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Wrapper de conveniencia que lee la estrategia desde la configuración
    y delega en impute_missing_values.

    Args:
        df: DataFrame a imputar
        config: Diccionario de configuración completo. Se espera la clave
                config['data']['imputation']['strategy'].

    Returns:
        DataFrame con valores imputados
    """
    strategy: str = (
        config.get('data', {})
              .get('imputation', {})
              .get('strategy', 'median')
    )
    return impute_missing_values(df, strategy=strategy)
