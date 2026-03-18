"""Clases de datos para el sistema de análisis de productividad estudiantil."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class StudentData:
    """Datos de un estudiante individual."""
    student_id: int
    age: int
    gender: str
    
    # Hábitos digitales
    phone_usage_hours: float
    social_media_hours: float
    youtube_hours: float
    gaming_hours: float
    
    # Hábitos académicos
    study_hours_per_day: float
    assignments_completed: int
    attendance_percentage: float
    
    # Estilo de vida
    sleep_hours: float
    exercise_minutes: int
    coffee_intake_mg: int
    breaks_per_day: int
    stress_level: int
    focus_score: int
    
    # Resultados
    final_grade: float
    productivity_score: float


@dataclass
class CorrelationResult:
    """Resultado de análisis de correlación."""
    feature: str
    target: str
    correlation: float
    p_value: float
    is_significant: bool
    confidence_level: float


@dataclass
class ModelEvaluation:
    """Métricas de evaluación de modelo."""
    model_name: str
    model_type: str
    
    # Métricas de error
    rmse: float
    mae: float
    r_squared: float
    
    # Validación cruzada
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Metadatos
    training_date: datetime
    n_samples_train: int
    n_samples_test: int
    hyperparameters: Dict[str, Any]


@dataclass
class FeatureImportance:
    """Importancia de una feature."""
    feature_name: str
    importance: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    rank: int


@dataclass
class Recommendation:
    """Recomendación personalizada para estudiante."""
    recommendation_id: str
    student_id: int
    category: str  # 'digital_habits', 'study_habits', 'lifestyle'
    description: str
    current_value: float
    target_value: float
    estimated_impact: float  # Mejora esperada en productivity_score
    priority: int  # 1 (alta) a 3 (baja)
    confidence: float  # 0.0 a 1.0


@dataclass
class ValidationResult:
    """Resultado de validación de datos."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_metrics: Dict[str, Any]
