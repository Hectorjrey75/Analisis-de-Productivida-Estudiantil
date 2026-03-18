"""
Módulo de validación de configuración.

Valida que los archivos de configuración YAML contengan valores válidos
y retorna errores específicos cuando se detectan problemas.
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Error de validación con ubicación y mensaje específico."""
    path: str
    message: str
    value: Any = None


class ConfigValidator:
    """Validador de configuración del sistema."""
    
    VALID_IMPUTATION_STRATEGIES = {"mean", "median", "mode", "forward_fill", "drop"}
    VALID_NORMALIZATION_METHODS = {"standard", "minmax"}
    VALID_ENCODING_METHODS = {"onehot", "label"}
    VALID_OUTLIER_STRATEGIES = {"clip", "remove", "winsorize"}
    VALID_CORRELATION_METHODS = {"pearson", "spearman", "kendall"}
    VALID_MODEL_TYPES = {"linear_regression", "random_forest", "gradient_boosting"}
    VALID_EXPORT_FORMATS = {"pickle", "onnx", "joblib"}
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Valida la configuración completa.
        
        Args:
            config: Diccionario de configuración cargado desde YAML
            
        Returns:
            Tupla (is_valid, errors) donde is_valid es True si no hay errores
        """
        self.errors = []
        
        # Validar secciones requeridas
        self._validate_required_sections(config)
        
        # Validar sección de datos
        if "data" in config:
            self._validate_data_section(config["data"])
        
        # Validar sección de preprocesamiento
        if "preprocessing" in config:
            self._validate_preprocessing_section(config["preprocessing"])
        
        # Validar sección de correlación
        if "correlation" in config:
            self._validate_correlation_section(config["correlation"])
        
        # Validar sección de modelos
        if "models" in config:
            self._validate_models_section(config["models"])
        
        # Validar sección de entrenamiento
        if "training" in config:
            self._validate_training_section(config["training"])
        
        # Validar sección de recomendaciones
        if "recommendations" in config:
            self._validate_recommendations_section(config["recommendations"])
        
        # Validar sección de exportación
        if "export" in config:
            self._validate_export_section(config["export"])
        
        return len(self.errors) == 0, self.errors
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> None:
        """Valida que existan las secciones requeridas."""
        required_sections = ["project", "data"]
        for section in required_sections:
            if section not in config:
                self.errors.append(
                    ValidationError(
                        path=section,
                        message=f"Sección requerida '{section}' no encontrada en configuración"
                    )
                )
    
    def _validate_data_section(self, data_config: Dict[str, Any]) -> None:
        """Valida la sección de datos."""
        # Validar paths requeridos
        if "raw_path" not in data_config:
            self.errors.append(
                ValidationError(
                    path="data.raw_path",
                    message="Campo requerido 'raw_path' no encontrado"
                )
            )
        
        # Validar columnas requeridas
        if "required_columns" in data_config:
            if not isinstance(data_config["required_columns"], list):
                self.errors.append(
                    ValidationError(
                        path="data.required_columns",
                        message="'required_columns' debe ser una lista",
                        value=data_config["required_columns"]
                    )
                )
            elif len(data_config["required_columns"]) == 0:
                self.errors.append(
                    ValidationError(
                        path="data.required_columns",
                        message="'required_columns' no puede estar vacía"
                    )
                )
        
        # Validar estrategia de imputación
        if "imputation" in data_config:
            imputation = data_config["imputation"]
            if "strategy" in imputation:
                strategy = imputation["strategy"]
                if strategy not in self.VALID_IMPUTATION_STRATEGIES:
                    self.errors.append(
                        ValidationError(
                            path="data.imputation.strategy",
                            message=f"Estrategia de imputación inválida: '{strategy}'. "
                                    f"Valores válidos: {self.VALID_IMPUTATION_STRATEGIES}",
                            value=strategy
                        )
                    )
    
    def _validate_preprocessing_section(self, preprocessing_config: Dict[str, Any]) -> None:
        """Valida la sección de preprocesamiento."""
        # Validar normalización
        if "normalization" in preprocessing_config:
            norm = preprocessing_config["normalization"]
            if "method" in norm:
                method = norm["method"]
                if method not in self.VALID_NORMALIZATION_METHODS:
                    self.errors.append(
                        ValidationError(
                            path="preprocessing.normalization.method",
                            message=f"Método de normalización inválido: '{method}'. "
                                    f"Valores válidos: {self.VALID_NORMALIZATION_METHODS}",
                            value=method
                        )
                    )
        
        # Validar codificación
        if "encoding" in preprocessing_config:
            enc = preprocessing_config["encoding"]
            if "method" in enc:
                method = enc["method"]
                if method not in self.VALID_ENCODING_METHODS:
                    self.errors.append(
                        ValidationError(
                            path="preprocessing.encoding.method",
                            message=f"Método de codificación inválido: '{method}'. "
                                    f"Valores válidos: {self.VALID_ENCODING_METHODS}",
                            value=method
                        )
                    )
        
        # Validar manejo de outliers
        if "outliers" in preprocessing_config:
            outliers = preprocessing_config["outliers"]
            if "strategy" in outliers:
                strategy = outliers["strategy"]
                if strategy not in self.VALID_OUTLIER_STRATEGIES:
                    self.errors.append(
                        ValidationError(
                            path="preprocessing.outliers.strategy",
                            message=f"Estrategia de outliers inválida: '{strategy}'. "
                                    f"Valores válidos: {self.VALID_OUTLIER_STRATEGIES}",
                            value=strategy
                        )
                    )
            
            if "threshold" in outliers:
                threshold = outliers["threshold"]
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    self.errors.append(
                        ValidationError(
                            path="preprocessing.outliers.threshold",
                            message=f"Threshold debe ser un número positivo, recibido: {threshold}",
                            value=threshold
                        )
                    )
    
    def _validate_correlation_section(self, correlation_config: Dict[str, Any]) -> None:
        """Valida la sección de correlación."""
        if "method" in correlation_config:
            method = correlation_config["method"]
            if method not in self.VALID_CORRELATION_METHODS:
                self.errors.append(
                    ValidationError(
                        path="correlation.method",
                        message=f"Método de correlación inválido: '{method}'. "
                                f"Valores válidos: {self.VALID_CORRELATION_METHODS}",
                        value=method
                    )
                )
        
        if "significance_level" in correlation_config:
            level = correlation_config["significance_level"]
            if not isinstance(level, (int, float)) or not (0 < level < 1):
                self.errors.append(
                    ValidationError(
                        path="correlation.significance_level",
                        message=f"Nivel de significancia debe estar entre 0 y 1, recibido: {level}",
                        value=level
                    )
                )
        
        if "multicollinearity_threshold" in correlation_config:
            threshold = correlation_config["multicollinearity_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                self.errors.append(
                    ValidationError(
                        path="correlation.multicollinearity_threshold",
                        message=f"Threshold de multicolinealidad debe estar entre 0 y 1, recibido: {threshold}",
                        value=threshold
                    )
                )
    
    def _validate_models_section(self, models_config: Dict[str, Any]) -> None:
        """Valida la sección de modelos."""
        if "types" in models_config:
            types = models_config["types"]
            if not isinstance(types, list):
                self.errors.append(
                    ValidationError(
                        path="models.types",
                        message="'types' debe ser una lista",
                        value=types
                    )
                )
            else:
                for model_type in types:
                    if model_type not in self.VALID_MODEL_TYPES:
                        self.errors.append(
                            ValidationError(
                                path="models.types",
                                message=f"Tipo de modelo inválido: '{model_type}'. "
                                        f"Valores válidos: {self.VALID_MODEL_TYPES}",
                                value=model_type
                            )
                        )
        
        if "production_threshold" in models_config:
            threshold = models_config["production_threshold"]
            if "r_squared" in threshold:
                r2 = threshold["r_squared"]
                if not isinstance(r2, (int, float)) or not (0 <= r2 <= 1):
                    self.errors.append(
                        ValidationError(
                            path="models.production_threshold.r_squared",
                            message=f"R² threshold debe estar entre 0 y 1, recibido: {r2}",
                            value=r2
                        )
                    )
    
    def _validate_training_section(self, training_config: Dict[str, Any]) -> None:
        """Valida la sección de entrenamiento."""
        # Validar test_size
        if "test_size" in training_config:
            test_size = training_config["test_size"]
            if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
                self.errors.append(
                    ValidationError(
                        path="training.test_size",
                        message=f"test_size debe estar entre 0 y 1, recibido: {test_size}",
                        value=test_size
                    )
                )
        
        # Validar validation_size
        if "validation_size" in training_config:
            val_size = training_config["validation_size"]
            if not isinstance(val_size, (int, float)) or not (0 < val_size < 1):
                self.errors.append(
                    ValidationError(
                        path="training.validation_size",
                        message=f"validation_size debe estar entre 0 y 1, recibido: {val_size}",
                        value=val_size
                    )
                )
        
        # Validar que test_size + validation_size < 1
        if "test_size" in training_config and "validation_size" in training_config:
            total = training_config["test_size"] + training_config["validation_size"]
            if total >= 1:
                self.errors.append(
                    ValidationError(
                        path="training",
                        message=f"test_size + validation_size debe ser menor a 1, suma actual: {total}",
                        value={"test_size": training_config["test_size"], 
                               "validation_size": training_config["validation_size"]}
                    )
                )
        
        # Validar cv_folds
        if "cv_folds" in training_config:
            cv_folds = training_config["cv_folds"]
            if not isinstance(cv_folds, int) or cv_folds < 2:
                self.errors.append(
                    ValidationError(
                        path="training.cv_folds",
                        message=f"cv_folds debe ser un entero >= 2, recibido: {cv_folds}",
                        value=cv_folds
                    )
                )
    
    def _validate_recommendations_section(self, recommendations_config: Dict[str, Any]) -> None:
        """Valida la sección de recomendaciones."""
        if "n_recommendations" in recommendations_config:
            n_rec = recommendations_config["n_recommendations"]
            if not isinstance(n_rec, int) or n_rec < 1:
                self.errors.append(
                    ValidationError(
                        path="recommendations.n_recommendations",
                        message=f"n_recommendations debe ser un entero positivo, recibido: {n_rec}",
                        value=n_rec
                    )
                )
        
        if "optimal_ranges" in recommendations_config:
            ranges = recommendations_config["optimal_ranges"]
            if not isinstance(ranges, dict):
                self.errors.append(
                    ValidationError(
                        path="recommendations.optimal_ranges",
                        message="optimal_ranges debe ser un diccionario",
                        value=ranges
                    )
                )
            else:
                for key, value in ranges.items():
                    if not isinstance(value, list) or len(value) != 2:
                        self.errors.append(
                            ValidationError(
                                path=f"recommendations.optimal_ranges.{key}",
                                message=f"Rango debe ser una lista de 2 elementos [min, max], recibido: {value}",
                                value=value
                            )
                        )
                    elif not all(isinstance(v, (int, float)) for v in value):
                        self.errors.append(
                            ValidationError(
                                path=f"recommendations.optimal_ranges.{key}",
                                message=f"Valores del rango deben ser numéricos, recibido: {value}",
                                value=value
                            )
                        )
                    elif value[0] >= value[1]:
                        self.errors.append(
                            ValidationError(
                                path=f"recommendations.optimal_ranges.{key}",
                                message=f"Valor mínimo debe ser menor que máximo, recibido: {value}",
                                value=value
                            )
                        )
    
    def _validate_export_section(self, export_config: Dict[str, Any]) -> None:
        """Valida la sección de exportación."""
        if "model_formats" in export_config:
            formats = export_config["model_formats"]
            if not isinstance(formats, list):
                self.errors.append(
                    ValidationError(
                        path="export.model_formats",
                        message="model_formats debe ser una lista",
                        value=formats
                    )
                )
            else:
                for fmt in formats:
                    if fmt not in self.VALID_EXPORT_FORMATS:
                        self.errors.append(
                            ValidationError(
                                path="export.model_formats",
                                message=f"Formato de exportación inválido: '{fmt}'. "
                                        f"Valores válidos: {self.VALID_EXPORT_FORMATS}",
                                value=fmt
                            )
                        )


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
    """
    Función de conveniencia para validar configuración.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Tupla (is_valid, errors)
    """
    validator = ConfigValidator()
    return validator.validate(config)
