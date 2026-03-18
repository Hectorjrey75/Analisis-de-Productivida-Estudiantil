"""
Pipeline principal del sistema de análisis de productividad estudiantil.

Ejecuta el pipeline completo o etapas seleccionadas:
  data → preprocess → features → correlation → train → recommend → visualize → export
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(log_file: str = "pipeline.log") -> logging.Logger:
    """Configura logging a consola y archivo."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Carga configuración desde archivo YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _stage(logger: logging.Logger, name: str):
    """Context manager simple para loguear inicio/fin y tiempo de una etapa."""
    class _Stage:
        def __enter__(self):
            logger.info(f"=== Iniciando etapa: {name} ===")
            self._start = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self._start
            if exc_type:
                logger.error(f"=== Etapa '{name}' falló en {elapsed:.2f}s: {exc_val} ===")
            else:
                logger.info(f"=== Etapa '{name}' completada en {elapsed:.2f}s ===")
            return False  # no suprimir excepciones

    return _Stage()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_data(config: Dict, logger: logging.Logger) -> Any:
    """Etapa 1: Ingesta y validación de datos."""
    from src.data.make_dataset import (
        compute_data_quality_stats,
        load_raw_data,
        validate_schema,
    )

    with _stage(logger, "data"):
        raw_path = config["data"]["raw_path"]
        df = load_raw_data(raw_path, config)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

        required_cols = config["data"].get("required_columns", [])
        validation = validate_schema(df, required_cols)
        if not validation.is_valid:
            logger.warning(f"Problemas de esquema: {validation.errors}")
        else:
            logger.info("Esquema validado correctamente")

        stats = compute_data_quality_stats(df)
        logger.info(
            f"Calidad de datos — duplicados: {stats['duplicate_rows']}, "
            f"filas totales: {stats['total_rows']}"
        )
        return df


def stage_preprocess(df: Any, config: Dict, logger: logging.Logger) -> Any:
    """Etapa 2: Preprocesamiento."""
    from src.data.preprocessing import DataPreprocessor

    with _stage(logger, "preprocess"):
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.fit_transform(df)
        logger.info(f"Preprocesamiento completado: {df_processed.shape}")
        return df_processed, preprocessor


def stage_features(df: Any, logger: logging.Logger) -> Any:
    """Etapa 3: Ingeniería de features."""
    from src.features.build_features import (
        create_derived_features,
        handle_outliers,
    )

    with _stage(logger, "features"):
        df_feat = create_derived_features(df)
        new_cols = set(df_feat.columns) - set(df.columns)
        logger.info(f"Features derivadas creadas: {sorted(new_cols)}")

        df_feat = handle_outliers(df_feat, strategy="clip", threshold=3.0)
        logger.info("Outliers manejados con estrategia 'clip'")
        return df_feat


def stage_correlation(df: Any, config: Dict, logger: logging.Logger) -> Any:
    """Etapa 4: Análisis de correlación."""
    from src.analysis.correlation import (
        compute_correlation_matrix,
        find_significant_correlations,
    )

    with _stage(logger, "correlation"):
        method = config.get("correlation", {}).get("method", "pearson")
        corr_matrix = compute_correlation_matrix(df, method=method)
        logger.info(f"Matriz de correlación calculada ({method}): {corr_matrix.shape}")

        target = "productivity_score"
        if target in corr_matrix.columns:
            sig_corr = find_significant_correlations(
                corr_matrix,
                target=target,
                confidence_level=config.get("correlation", {}).get("significance_level", 0.95),
                n_samples=len(df),
            )
            logger.info(f"Correlaciones significativas con '{target}': {len(sig_corr)}")
        else:
            sig_corr = None
            logger.warning(f"Variable objetivo '{target}' no encontrada para correlación")

        return corr_matrix, sig_corr


def stage_train(df: Any, config: Dict, logger: logging.Logger) -> Any:
    """Etapa 5: Entrenamiento de modelos."""
    import pandas as pd
    from src.models.train_model import ModelTrainer
    from src.models.predict_model import extract_feature_importance

    with _stage(logger, "train"):
        target_col = "productivity_score"
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el DataFrame")

        numeric_df = df.select_dtypes(include="number")
        feature_cols = [c for c in numeric_df.columns if c != target_col]
        X = numeric_df[feature_cols].fillna(0)
        y = numeric_df[target_col].fillna(0)

        trainer = ModelTrainer(config)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)

        best_model = None
        best_r2 = -float("inf")
        best_name = None
        all_results = {}

        for model_type in config.get("models", {}).get("types", ["random_forest"]):
            try:
                model = trainer.train_model(model_type, X_train, y_train)
                metrics = trainer.evaluate_model(model, X_test, y_test)
                all_results[model_type] = metrics
                logger.info(
                    f"Modelo '{model_type}': R²={metrics['r_squared']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
                )
                if metrics["r_squared"] > best_r2:
                    best_r2 = metrics["r_squared"]
                    best_model = model
                    best_name = model_type
            except Exception as exc:
                logger.error(f"Error entrenando '{model_type}': {exc}")

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningún modelo")

        logger.info(f"Mejor modelo: '{best_name}' con R²={best_r2:.4f}")

        importance_df = extract_feature_importance(best_model, feature_cols)
        logger.info(f"Top 3 features: {importance_df['feature'].head(3).tolist()}")

        return best_model, importance_df, X_test, y_test, all_results, feature_cols


def stage_recommend(
    df: Any,
    model: Any,
    importance_df: Any,
    config: Dict,
    logger: logging.Logger,
    feature_cols: Optional[List[str]] = None,
) -> Any:
    """Etapa 6: Generación de recomendaciones."""
    from src.recommendations.generator import RecommendationEngine

    with _stage(logger, "recommend"):
        optimal_ranges = config.get("recommendations", {}).get("optimal_ranges", {})
        n_recs = config.get("recommendations", {}).get("n_recommendations", 3)

        engine = RecommendationEngine(model, importance_df, optimal_ranges)

        # Use only the feature columns the model was trained on
        if feature_cols:
            available = [c for c in feature_cols if c in df.columns]
            sample_df = df[available]
        else:
            sample_df = df.select_dtypes(include="number")

        if sample_df.empty:
            logger.warning("No hay datos numéricos para generar recomendaciones")
            return []

        sample_student = sample_df.iloc[0]
        recommendations = engine.generate_recommendations(sample_student, n_recommendations=n_recs)
        logger.info(f"Recomendaciones generadas: {len(recommendations)}")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"  Rec {i} [{rec['feature']}]: {rec['description'][:80]}..."
                if len(rec["description"]) > 80
                else f"  Rec {i} [{rec['feature']}]: {rec['description']}"
            )
        return recommendations


def stage_visualize(corr_matrix: Any, importance_df: Any, X_test: Any, y_test: Any,
                    model: Any, metrics: Dict, output_dir: str, logger: logging.Logger) -> None:
    """Etapa 7: Visualización."""
    import numpy as np
    from src.visualization.visualize import (
        create_dashboard,
        plot_correlation_matrix,
        plot_feature_importance,
    )

    with _stage(logger, "visualize"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Correlation heatmap
        corr_path = str(out / "correlation_matrix.png")
        plot_correlation_matrix(corr_matrix, output_path=corr_path)
        logger.info(f"Heatmap de correlación guardado: {corr_path}")

        # Feature importance
        imp_path = str(out / "feature_importance.png")
        plot_feature_importance(importance_df, output_path=imp_path)
        logger.info(f"Gráfico de importancia guardado: {imp_path}")

        # Dashboard
        y_pred = model.predict(X_test)
        dashboard_results = {
            "corr_matrix": corr_matrix,
            "feature_importance": importance_df,
            "y_true": y_test.values if hasattr(y_test, "values") else y_test,
            "y_pred": y_pred,
            "metrics": metrics,
        }
        dash_path = str(out / "dashboard.png")
        create_dashboard(dashboard_results, output_path=dash_path)
        logger.info(f"Dashboard guardado: {dash_path}")


def stage_export(df: Any, model: Any, recommendations: Any, metrics: Dict,
                 config: Dict, output_dir: str, logger: logging.Logger) -> None:
    """Etapa 8: Exportación de resultados."""
    from src.export.exporter import (
        compute_checksum,
        export_dataframe,
        export_results_with_metadata,
    )

    with _stage(logger, "export"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Export processed dataset
        csv_path = str(out / "processed_data.csv")
        export_dataframe(df, csv_path, format="csv")
        checksum = compute_checksum(csv_path)
        logger.info(f"Dataset exportado: {csv_path} (SHA256: {checksum[:16]}...)")

        # Export results with metadata
        results_path = str(out / "results.json")
        results = {
            "model_metrics": metrics,
            "n_recommendations": len(recommendations) if recommendations else 0,
        }
        metadata = {
            "version": config.get("project", {}).get("version", "1.0.0"),
            "config": config.get("project", {}),
        }
        export_results_with_metadata(results, results_path, metadata)
        logger.info(f"Resultados exportados: {results_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

ALL_STAGES = ["data", "preprocess", "features", "correlation", "train", "recommend", "visualize", "export"]


def run_pipeline(
    config: Dict[str, Any],
    stages: Optional[List[str]] = None,
    output_dir: str = "data/processed",
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo o las etapas indicadas.

    Args:
        config: Diccionario de configuración cargado desde YAML.
        stages: Lista de etapas a ejecutar. None ejecuta todas.
        output_dir: Directorio de salida para artefactos.

    Returns:
        Diccionario con los artefactos producidos por cada etapa.
    """
    logger = configure_logging()
    stages_to_run = stages if stages else ALL_STAGES

    logger.info(f"Pipeline iniciado. Etapas: {stages_to_run}")
    pipeline_start = time.time()

    artifacts: Dict[str, Any] = {}

    try:
        if "data" in stages_to_run:
            artifacts["df_raw"] = stage_data(config, logger)

        if "preprocess" in stages_to_run:
            df_input = artifacts.get("df_raw")
            if df_input is None:
                raise RuntimeError("Etapa 'data' requerida antes de 'preprocess'")
            artifacts["df_processed"], artifacts["preprocessor"] = stage_preprocess(df_input, config, logger)

        if "features" in stages_to_run:
            df_input = artifacts.get("df_processed", artifacts.get("df_raw"))
            if df_input is None:
                raise RuntimeError("Se requiere un DataFrame para la etapa 'features'")
            artifacts["df_features"] = stage_features(df_input, logger)

        if "correlation" in stages_to_run:
            df_input = artifacts.get("df_features", artifacts.get("df_processed", artifacts.get("df_raw")))
            if df_input is None:
                raise RuntimeError("Se requiere un DataFrame para la etapa 'correlation'")
            artifacts["corr_matrix"], artifacts["sig_corr"] = stage_correlation(df_input, config, logger)

        if "train" in stages_to_run:
            df_input = artifacts.get("df_features", artifacts.get("df_processed", artifacts.get("df_raw")))
            if df_input is None:
                raise RuntimeError("Se requiere un DataFrame para la etapa 'train'")
            (
                artifacts["model"],
                artifacts["importance_df"],
                artifacts["X_test"],
                artifacts["y_test"],
                artifacts["model_metrics"],
                artifacts["feature_cols"],
            ) = stage_train(df_input, config, logger)

        if "recommend" in stages_to_run:
            df_input = artifacts.get("df_features", artifacts.get("df_processed", artifacts.get("df_raw")))
            model = artifacts.get("model")
            importance_df = artifacts.get("importance_df")
            feature_cols = artifacts.get("feature_cols")
            if df_input is None or model is None or importance_df is None:
                logger.warning("Saltando etapa 'recommend': faltan artefactos de entrenamiento")
            else:
                artifacts["recommendations"] = stage_recommend(
                    df_input, model, importance_df, config, logger, feature_cols=feature_cols
                )

        if "visualize" in stages_to_run:
            corr_matrix = artifacts.get("corr_matrix")
            importance_df = artifacts.get("importance_df")
            model = artifacts.get("model")
            X_test = artifacts.get("X_test")
            y_test = artifacts.get("y_test")
            metrics = artifacts.get("model_metrics", {})
            if any(v is None for v in [corr_matrix, importance_df, model, X_test, y_test]):
                logger.warning("Saltando etapa 'visualize': faltan artefactos previos")
            else:
                best_metrics = max(metrics.values(), key=lambda m: m.get("r_squared", -1)) if metrics else {}
                stage_visualize(corr_matrix, importance_df, X_test, y_test, model, best_metrics, output_dir, logger)

        if "export" in stages_to_run:
            df_input = artifacts.get("df_features", artifacts.get("df_processed", artifacts.get("df_raw")))
            model = artifacts.get("model")
            recommendations = artifacts.get("recommendations", [])
            metrics = artifacts.get("model_metrics", {})
            if df_input is None:
                logger.warning("Saltando etapa 'export': falta DataFrame")
            else:
                best_metrics = max(metrics.values(), key=lambda m: m.get("r_squared", -1)) if metrics else {}
                stage_export(df_input, model, recommendations, best_metrics, config, output_dir, logger)

    except Exception as exc:
        logger.error(f"Pipeline falló: {exc}", exc_info=True)
        raise

    elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline completado en {elapsed:.2f}s")
    return artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis de productividad estudiantil",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Etapas disponibles: data, preprocess, features, correlation, "
            "train, recommend, visualize, export\n\n"
            "Ejemplo:\n"
            "  python pipeline.py --stages data,preprocess,features,train"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuración YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help=(
            "Etapas a ejecutar separadas por coma "
            "(default: todas). Ej: data,preprocess,train"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        dest="output_dir",
        help="Directorio de salida para resultados (default: data/processed)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stages: Optional[List[str]] = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        invalid = [s for s in stages if s not in ALL_STAGES]
        if invalid:
            print(f"Error: etapas no válidas: {invalid}. Válidas: {ALL_STAGES}", file=sys.stderr)
            return 1

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        run_pipeline(config, stages=stages, output_dir=args.output_dir)
    except Exception as exc:
        print(f"Error en pipeline: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
