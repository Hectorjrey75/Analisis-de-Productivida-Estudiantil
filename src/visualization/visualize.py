"""
Módulo de visualización para el sistema de análisis de productividad estudiantil.
Genera gráficos, dashboards y visualizaciones de resultados.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, Optional


# Configuración por defecto de visualización
_DEFAULT_STYLE = "seaborn-v0_8-darkgrid"
_DEFAULT_PALETTE = "husl"
_DEFAULT_FIGSIZE = (12, 8)
_DEFAULT_DPI = 300


def _apply_style() -> None:
    """Aplica el estilo y paleta de configuración."""
    try:
        plt.style.use(_DEFAULT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette(_DEFAULT_PALETTE)


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: str = None
) -> None:
    """
    Genera heatmap de matriz de correlación.

    Args:
        corr_matrix: DataFrame con la matriz de correlación.
        output_path: Ruta opcional para guardar la figura.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriz de Correlación", fontsize=14, pad=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_scatter_with_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str = None
) -> None:
    """
    Genera gráfico de dispersión con línea de regresión.

    Args:
        df: DataFrame con los datos.
        x: Nombre de la columna para el eje X.
        y: Nombre de la columna para el eje Y.
        output_path: Ruta opcional para guardar la figura.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    sns.regplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": 0.5, "s": 30},
        line_kws={"color": "red", "linewidth": 2},
    )
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(f"{y} vs {x}", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    output_path: str = None
) -> None:
    """
    Genera gráfico de barras horizontal de importancia de features.

    Args:
        importance_df: DataFrame con columnas 'feature' e 'importance'.
                       Opcionalmente puede incluir 'ci_lower' y 'ci_upper'
                       para intervalos de confianza.
        top_n: Número de features a mostrar.
        output_path: Ruta opcional para guardar la figura.
    """
    _apply_style()

    df = importance_df.copy()
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError("importance_df debe contener columnas 'feature' e 'importance'.")

    df = df.nlargest(top_n, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    has_ci = "ci_lower" in df.columns and "ci_upper" in df.columns
    if has_ci:
        xerr = [
            df["importance"] - df["ci_lower"],
            df["ci_upper"] - df["importance"],
        ]
        ax.barh(df["feature"], df["importance"], xerr=xerr, capsize=4, color="steelblue")
    else:
        ax.barh(df["feature"], df["importance"], color="steelblue")

    ax.set_xlabel("Importancia", fontsize=12)
    ax.set_title(f"Top {top_n} Features más Importantes", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = None
) -> None:
    """
    Genera gráfico de predicciones vs valores reales.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        output_path: Ruta opcional para guardar la figura.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, label="Predicciones")

    # Línea diagonal de referencia (predicción perfecta)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Predicción perfecta")

    ax.set_xlabel("Valores Reales", fontsize=12)
    ax.set_ylabel("Predicciones", fontsize=12)
    ax.set_title("Predicciones vs Valores Reales", fontsize=14)
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = None
) -> None:
    """
    Genera gráfico de residuos del modelo.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        output_path: Ruta opcional para guardar la figura.
    """
    _apply_style()
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    ax.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Predicciones", fontsize=12)
    ax.set_ylabel("Residuos", fontsize=12)
    ax.set_title("Residuos vs Predicciones", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")

    plt.close(fig)


def create_dashboard(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Crea dashboard con métricas principales, correlaciones e importancia de features.

    Args:
        results: Diccionario con:
            - 'corr_matrix': DataFrame con matriz de correlación
            - 'feature_importance': DataFrame con importancia de features
            - 'y_true': array de valores reales
            - 'y_pred': array de predicciones
            - 'metrics': dict con rmse, mae, r_squared
        output_path: Ruta para guardar el dashboard.
    """
    _apply_style()

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: Métricas principales (texto) ---
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis("off")
    metrics = results.get("metrics", {})
    rmse = metrics.get("rmse", float("nan"))
    mae = metrics.get("mae", float("nan"))
    r2 = metrics.get("r_squared", float("nan"))
    metrics_text = (
        "Métricas del Modelo\n"
        "─────────────────────\n"
        f"RMSE:      {rmse:.4f}\n"
        f"MAE:       {mae:.4f}\n"
        f"R²:        {r2:.4f}"
    )
    ax_metrics.text(
        0.1, 0.5, metrics_text,
        transform=ax_metrics.transAxes,
        fontsize=13,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4),
    )
    ax_metrics.set_title("Métricas Principales", fontsize=13)

    # --- Panel 2: Heatmap de correlación ---
    ax_corr = fig.add_subplot(gs[0, 1])
    corr_matrix = results.get("corr_matrix")
    if corr_matrix is not None and not corr_matrix.empty:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=len(corr_matrix) <= 10,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.3,
            ax=ax_corr,
            cbar_kws={"shrink": 0.8},
        )
    ax_corr.set_title("Matriz de Correlación", fontsize=13)

    # --- Panel 3: Importancia de features ---
    ax_imp = fig.add_subplot(gs[1, 0])
    importance_df = results.get("feature_importance")
    if importance_df is not None and not importance_df.empty:
        top = importance_df.nlargest(10, "importance").sort_values("importance")
        ax_imp.barh(top["feature"], top["importance"], color="steelblue")
        ax_imp.set_xlabel("Importancia", fontsize=11)
    ax_imp.set_title("Top 10 Features Importantes", fontsize=13)

    # --- Panel 4: Predicciones vs Reales ---
    ax_pred = fig.add_subplot(gs[1, 1])
    y_true = results.get("y_true")
    y_pred = results.get("y_pred")
    if y_true is not None and y_pred is not None:
        ax_pred.scatter(y_true, y_pred, alpha=0.4, s=20)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax_pred.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax_pred.set_xlabel("Valores Reales", fontsize=11)
        ax_pred.set_ylabel("Predicciones", fontsize=11)
    ax_pred.set_title("Predicciones vs Reales", fontsize=13)

    fig.suptitle("Dashboard de Análisis de Productividad Estudiantil", fontsize=16, y=1.01)

    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
