"""
Módulo de generación de recomendaciones personalizadas.
Implementa RecommendationEngine para analizar perfiles estudiantiles
y generar recomendaciones basadas en rangos óptimos y modelos predictivos.
"""

from typing import Any, Dict, List

import pandas as pd


# Textos de recomendación por feature
_RECOMMENDATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "study_hours_per_day": {
        "below": (
            "Incrementa tus horas de estudio diarias a al menos {target:.1f} horas. "
            "Actualmente estudias {current:.1f} h/día, lo que está por debajo del rango óptimo."
        ),
        "above": (
            "Considera reducir tus horas de estudio a {target:.1f} h/día para evitar el agotamiento. "
            "Actualmente estudias {current:.1f} h/día."
        ),
    },
    "sleep_hours": {
        "below": (
            "Aumenta tus horas de sueño a al menos {target:.1f} horas por noche. "
            "Dormir {current:.1f} h es insuficiente para una recuperación óptima."
        ),
        "above": (
            "Intenta reducir el tiempo de sueño a {target:.1f} h para mantener un ritmo activo. "
            "Actualmente duermes {current:.1f} h."
        ),
    },
    "phone_usage_hours": {
        "below": (
            "Tu uso del teléfono ({current:.1f} h/día) está dentro del rango óptimo."
        ),
        "above": (
            "Reduce el uso del teléfono a menos de {target:.1f} h/día. "
            "Actualmente usas el teléfono {current:.1f} h/día, lo que puede distraerte."
        ),
    },
    "social_media_hours": {
        "below": (
            "Tu uso de redes sociales ({current:.1f} h/día) está dentro del rango óptimo."
        ),
        "above": (
            "Limita el uso de redes sociales a {target:.1f} h/día. "
            "Actualmente dedicas {current:.1f} h/día, lo que reduce tu tiempo productivo."
        ),
    },
    "exercise_minutes": {
        "below": (
            "Aumenta tu actividad física a al menos {target:.0f} minutos diarios. "
            "Actualmente ejercitas {current:.0f} min/día, por debajo del mínimo recomendado."
        ),
        "above": (
            "Tu nivel de ejercicio ({current:.0f} min/día) supera el rango óptimo; "
            "asegúrate de no sobreentrenarte."
        ),
    },
}

_DEFAULT_TEMPLATE = (
    "Ajusta '{feature}' de {current:.2f} a {target:.2f} para mejorar tu productividad."
)


def _build_description(feature: str, direction: str, current: float, target: float) -> str:
    """Construye texto de recomendación para una feature y dirección dadas."""
    templates = _RECOMMENDATION_TEMPLATES.get(feature, {})
    template = templates.get(direction, _DEFAULT_TEMPLATE)
    return template.format(feature=feature, current=current, target=target)


class RecommendationEngine:
    """Motor de generación de recomendaciones personalizadas."""

    def __init__(
        self,
        model: Any,
        feature_importance: pd.DataFrame,
        optimal_ranges: Dict[str, List[float]],
    ):
        """
        Inicializa el motor de recomendaciones.

        Args:
            model: Modelo predictivo entrenado con método predict().
            feature_importance: DataFrame con columnas ['feature', 'importance']
                                 ordenado por importancia descendente.
            optimal_ranges: Diccionario {feature: [min, max]} con rangos óptimos.
                            Puede cargarse desde config['recommendations']['optimal_ranges'].
        """
        self.model = model
        self.feature_importance = feature_importance.copy()
        self.optimal_ranges = optimal_ranges

        # Índice de importancia por feature para acceso rápido
        self._importance_index: Dict[str, float] = {}
        if "feature" in feature_importance.columns and "importance" in feature_importance.columns:
            self._importance_index = dict(
                zip(feature_importance["feature"], feature_importance["importance"])
            )

    # ------------------------------------------------------------------
    # 10.2 analyze_student_profile
    # ------------------------------------------------------------------

    def analyze_student_profile(self, student_data: pd.Series) -> Dict[str, Any]:
        """
        Analiza el perfil de un estudiante e identifica áreas de mejora.

        Compara los valores actuales del estudiante con los rangos óptimos
        configurados y calcula los gaps para cada feature.

        Args:
            student_data: Serie de pandas con los valores del estudiante.
                          Los índices deben coincidir con las features del modelo.

        Returns:
            Diccionario con:
                - gaps (Dict[str, float]): {feature: gap} donde gap = optimal_min - current
                  (solo para features por debajo del óptimo) o
                  current - optimal_max (para features por encima del óptimo).
                - below_optimal (List[str]): features con valor < optimal_min.
                - above_optimal (List[str]): features con valor > optimal_max.
        """
        below_optimal: List[str] = []
        above_optimal: List[str] = []
        gaps: Dict[str, float] = {}

        for feature, (opt_min, opt_max) in self.optimal_ranges.items():
            if feature not in student_data.index:
                continue

            current_value = float(student_data[feature])

            if current_value < opt_min:
                below_optimal.append(feature)
                gaps[feature] = opt_min - current_value
            elif current_value > opt_max:
                above_optimal.append(feature)
                gaps[feature] = current_value - opt_max

        return {
            "gaps": gaps,
            "below_optimal": below_optimal,
            "above_optimal": above_optimal,
        }

    # ------------------------------------------------------------------
    # 10.3 generate_recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        student_data: pd.Series,
        n_recommendations: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones personalizadas para un estudiante.

        Prioriza basándose en la importancia de features y el tamaño del gap.
        Genera al menos n_recommendations recomendaciones cuando hay suficientes
        features fuera del rango óptimo.

        Args:
            student_data: Serie de pandas con los valores del estudiante.
            n_recommendations: Número mínimo de recomendaciones a generar (default 3).

        Returns:
            Lista de diccionarios con:
                - feature (str): nombre de la feature.
                - description (str): texto de recomendación legible.
                - current_value (float): valor actual del estudiante.
                - target_value (float): valor objetivo dentro del rango óptimo.
                - estimated_impact (float): mejora estimada en productivity_score.
                - priority (int): 1 (alta), 2 (media) o 3 (baja).
        """
        profile = self.analyze_student_profile(student_data)
        gaps = profile["gaps"]
        below_optimal = set(profile["below_optimal"])
        above_optimal = set(profile["above_optimal"])

        if not gaps:
            return []

        # Calcular puntuación de prioridad: importancia * gap normalizado
        max_gap = max(gaps.values()) if gaps else 1.0

        scored: List[Dict[str, Any]] = []
        for feature, gap in gaps.items():
            importance = self._importance_index.get(feature, 0.0)
            score = importance * (gap / max_gap)

            opt_min, opt_max = self.optimal_ranges[feature]
            current_value = float(student_data[feature])

            if feature in below_optimal:
                target_value = opt_min
                direction = "below"
            else:
                target_value = opt_max
                direction = "above"

            description = _build_description(feature, direction, current_value, target_value)

            scored.append(
                {
                    "feature": feature,
                    "description": description,
                    "current_value": current_value,
                    "target_value": target_value,
                    "_score": score,
                    "_direction": direction,
                }
            )

        # Ordenar por puntuación descendente
        scored.sort(key=lambda x: x["_score"], reverse=True)

        # Asignar prioridades y calcular impacto
        recommendations: List[Dict[str, Any]] = []
        n = len(scored)

        for i, item in enumerate(scored[:n_recommendations]):
            # Prioridad: top tercio → 1, segundo tercio → 2, resto → 3
            if n_recommendations <= 1:
                priority = 1
            elif i < n_recommendations // 3 or i == 0:
                priority = 1
            elif i < 2 * n_recommendations // 3:
                priority = 2
            else:
                priority = 3

            rec = {
                "feature": item["feature"],
                "description": item["description"],
                "current_value": item["current_value"],
                "target_value": item["target_value"],
                "estimated_impact": 0.0,  # se calcula a continuación
                "priority": priority,
            }

            # Calcular impacto estimado
            rec["estimated_impact"] = self.estimate_impact(student_data, rec)
            recommendations.append(rec)

        return recommendations

    # ------------------------------------------------------------------
    # 10.4 estimate_impact
    # ------------------------------------------------------------------

    def estimate_impact(
        self,
        student_data: pd.Series,
        recommendation: Dict[str, Any],
    ) -> float:
        """
        Estima el impacto de una recomendación en el productivity_score.

        Crea una copia del perfil del estudiante, aplica el cambio sugerido
        y predice el nuevo score con el modelo.

        Args:
            student_data: Serie de pandas con los valores actuales del estudiante.
            recommendation: Diccionario con al menos 'feature' y 'target_value'.

        Returns:
            Diferencia estimada en productivity_score (new_score - current_score).
            Un valor positivo indica mejora.
        """
        feature = recommendation["feature"]
        target_value = recommendation["target_value"]

        # Construir DataFrame con los datos actuales del estudiante
        student_df = student_data.to_frame().T.reset_index(drop=True)

        # Predicción con valores actuales
        current_score = float(self.model.predict(student_df)[0])

        # Aplicar cambio recomendado
        modified_df = student_df.copy()
        if feature in modified_df.columns:
            modified_df[feature] = target_value

        # Predicción con el cambio aplicado
        new_score = float(self.model.predict(modified_df)[0])

        return new_score - current_score
