"""
Módulo de monitoreo de modelos.
Implementa ModelMonitor para evaluar, detectar degradación y registrar historial de métricas.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelMonitor:
    """Monitor de rendimiento de modelos con detección de degradación."""

    def __init__(self, degradation_threshold: float = 0.10):
        """
        Inicializa el monitor con umbral de degradación.

        Args:
            degradation_threshold: Porcentaje de degradación permitido (default 10%).
        """
        self.degradation_threshold = degradation_threshold
        self.history: List[Dict] = []

    def evaluate_and_monitor(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict:
        """
        Evalúa el modelo y detecta degradación respecto al historial.

        Args:
            model: Modelo entrenado con método predict().
            X_test: Features de prueba.
            y_test: Target de prueba.
            model_name: Nombre identificador del modelo.

        Returns:
            Diccionario con métricas: rmse, mae, r_squared, model_name, timestamp.
        """
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r_squared = float(r2_score(y_test, y_pred))

        # Buscar última entrada histórica para este modelo
        previous = self._get_last_entry(model_name)
        if previous is not None:
            prev_r2 = previous["r_squared"]
            if prev_r2 != 0:
                degradation = (prev_r2 - r_squared) / abs(prev_r2)
                if degradation > self.degradation_threshold:
                    warnings.warn(
                        f"[ModelMonitor] Degradación detectada en '{model_name}': "
                        f"R² bajó de {prev_r2:.4f} a {r_squared:.4f} "
                        f"({degradation * 100:.1f}% > umbral {self.degradation_threshold * 100:.1f}%)",
                        UserWarning,
                        stacklevel=2,
                    )

        entry = {
            "model_name": model_name,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(entry)
        return entry

    def get_performance_history(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        Retorna el historial de métricas.

        Args:
            model_name: Si se proporciona, filtra por nombre de modelo.

        Returns:
            Lista de diccionarios con métricas e historial.
        """
        if model_name is None:
            return list(self.history)
        return [entry for entry in self.history if entry["model_name"] == model_name]

    def _get_last_entry(self, model_name: str) -> Optional[Dict]:
        """Retorna la última entrada histórica para el modelo dado."""
        entries = [e for e in self.history if e["model_name"] == model_name]
        return entries[-1] if entries else None
