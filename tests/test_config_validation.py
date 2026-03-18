"""
Tests de propiedad para validación de configuración.

**Validates: Requirements 9.4**
"""

import pytest
from hypothesis import given, strategies as st, assume
from config.validator import ConfigValidator, validate_config, ValidationError
import yaml
from pathlib import Path


# Estrategias para generar configuraciones válidas
valid_imputation_strategies = st.sampled_from(["mean", "median", "mode", "forward_fill", "drop"])
valid_normalization_methods = st.sampled_from(["standard", "minmax"])
valid_encoding_methods = st.sampled_from(["onehot", "label"])
valid_outlier_strategies = st.sampled_from(["clip", "remove", "winsorize"])
valid_correlation_methods = st.sampled_from(["pearson", "spearman", "kendall"])
valid_model_types = st.sampled_from(["linear_regression", "random_forest", "gradient_boosting"])
valid_export_formats = st.sampled_from(["pickle", "onnx", "joblib"])


@st.composite
def valid_config_strategy(draw):
    """
    Estrategia para generar configuraciones válidas.
    
    Esta estrategia genera configuraciones que deben pasar la validación
    sin errores, cumpliendo con todos los requisitos del sistema.
    """
    # Generar test_size y validation_size que sumen menos de 1
    test_size = draw(st.floats(min_value=0.05, max_value=0.3))
    validation_size = draw(st.floats(min_value=0.05, max_value=min(0.3, 0.95 - test_size)))
    
    # Generar rangos óptimos válidos
    def generate_valid_range():
        min_val = draw(st.floats(min_value=0.0, max_value=50.0))
        max_val = draw(st.floats(min_value=min_val + 0.1, max_value=min_val + 100.0))
        return [min_val, max_val]
    
    config = {
        "project": {
            "name": draw(st.text(min_size=1, max_size=50)),
            "version": draw(st.text(min_size=1, max_size=20)),
        },
        "data": {
            "raw_path": draw(st.text(min_size=1, max_size=100)),
            "processed_path": draw(st.text(min_size=1, max_size=100)),
            "required_columns": draw(st.lists(
                st.text(min_size=1, max_size=30),
                min_size=1,
                max_size=20
            )),
            "imputation": {
                "strategy": draw(valid_imputation_strategies)
            }
        },
        "preprocessing": {
            "normalization": {
                "method": draw(valid_normalization_methods)
            },
            "encoding": {
                "method": draw(valid_encoding_methods)
            },
            "outliers": {
                "strategy": draw(valid_outlier_strategies),
                "threshold": draw(st.floats(min_value=0.1, max_value=10.0))
            }
        },
        "correlation": {
            "method": draw(valid_correlation_methods),
            "significance_level": draw(st.floats(min_value=0.01, max_value=0.99)),
            "multicollinearity_threshold": draw(st.floats(min_value=0.0, max_value=1.0))
        },
        "models": {
            "types": draw(st.lists(
                valid_model_types,
                min_size=1,
                max_size=3,
                unique=True
            )),
            "production_threshold": {
                "r_squared": draw(st.floats(min_value=0.0, max_value=1.0))
            }
        },
        "training": {
            "test_size": test_size,
            "validation_size": validation_size,
            "random_state": draw(st.integers(min_value=0, max_value=10000)),
            "cv_folds": draw(st.integers(min_value=2, max_value=10))
        },
        "recommendations": {
            "n_recommendations": draw(st.integers(min_value=1, max_value=10)),
            "optimal_ranges": {
                "study_hours_per_day": generate_valid_range(),
                "sleep_hours": generate_valid_range(),
                "phone_usage_hours": generate_valid_range(),
            }
        },
        "export": {
            "model_formats": draw(st.lists(
                valid_export_formats,
                min_size=1,
                max_size=3,
                unique=True
            )),
            "include_metadata": draw(st.booleans()),
            "compute_checksums": draw(st.booleans())
        }
    }
    
    return config


class TestConfigurationValidation:
    """Tests de propiedad para validación de configuración."""
    
    @given(valid_config_strategy())
    def test_valid_configuration_loads_without_errors(self, config):
        """
        **Propiedad: Configuración válida debe cargarse sin errores**
        
        **Validates: Requirements 9.4**
        
        Para toda configuración que cumple con las especificaciones del sistema:
        - La validación debe retornar is_valid=True
        - La lista de errores debe estar vacía
        - No debe lanzar excepciones durante la validación
        """
        # Ejecutar validación
        is_valid, errors = validate_config(config)
        
        # Verificar que la configuración es válida
        assert is_valid, f"Configuración válida fue rechazada. Errores: {[e.message for e in errors]}"
        assert len(errors) == 0, f"Configuración válida generó errores: {errors}"
    
    def test_real_config_file_is_valid(self):
        """
        Test que verifica que el archivo de configuración real del proyecto es válido.
        
        **Validates: Requirements 9.4**
        """
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        is_valid, errors = validate_config(config)
        
        assert is_valid, f"Configuración del proyecto es inválida. Errores: {[e.message for e in errors]}"
        assert len(errors) == 0
    
    @given(st.dictionaries(st.text(), st.text()))
    def test_missing_required_sections_returns_errors(self, config):
        """
        **Propiedad: Configuraciones sin secciones requeridas deben retornar errores**
        
        **Validates: Requirements 9.4**
        
        Para toda configuración que no contiene las secciones requeridas:
        - La validación debe retornar is_valid=False
        - Debe haber al menos un error de validación
        - Los errores deben ser específicos y descriptivos
        """
        # Asegurar que no tiene las secciones requeridas
        assume("project" not in config or "data" not in config)
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, "Configuración inválida fue aceptada"
        assert len(errors) > 0, "No se generaron errores para configuración inválida"
        
        # Verificar que los errores son específicos
        error_messages = [e.message for e in errors]
        assert any("requerida" in msg.lower() or "required" in msg.lower() 
                   for msg in error_messages), "Errores no son específicos sobre secciones requeridas"
    
    @given(st.text().filter(lambda x: x not in ["mean", "median", "mode", "forward_fill", "drop"]))
    def test_invalid_imputation_strategy_returns_error(self, invalid_strategy):
        """
        **Propiedad: Estrategias de imputación inválidas deben retornar error específico**
        
        **Validates: Requirements 9.4**
        """
        config = {
            "project": {"name": "test"},
            "data": {
                "raw_path": "test.csv",
                "required_columns": ["col1"],
                "imputation": {"strategy": invalid_strategy}
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Estrategia inválida '{invalid_strategy}' fue aceptada"
        assert len(errors) > 0
        
        # Verificar que hay un error específico sobre la estrategia de imputación
        imputation_errors = [e for e in errors if "imputation" in e.path.lower()]
        assert len(imputation_errors) > 0, "No se generó error específico para estrategia de imputación inválida"
    
    @given(st.text().filter(lambda x: x not in ["standard", "minmax"]))
    def test_invalid_normalization_method_returns_error(self, invalid_method):
        """
        **Propiedad: Métodos de normalización inválidos deben retornar error específico**
        
        **Validates: Requirements 9.4**
        """
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "preprocessing": {
                "normalization": {"method": invalid_method}
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Método inválido '{invalid_method}' fue aceptado"
        
        # Verificar error específico
        norm_errors = [e for e in errors if "normalization" in e.path.lower()]
        assert len(norm_errors) > 0, "No se generó error específico para método de normalización inválido"
    
    @given(st.floats().filter(lambda x: x <= 0 or x >= 1))
    def test_invalid_significance_level_returns_error(self, invalid_level):
        """
        **Propiedad: Niveles de significancia fuera del rango (0, 1) deben retornar error**
        
        **Validates: Requirements 9.4**
        """
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "correlation": {
                "significance_level": invalid_level
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Nivel de significancia inválido {invalid_level} fue aceptado"
        
        # Verificar error específico
        sig_errors = [e for e in errors if "significance" in e.path.lower()]
        assert len(sig_errors) > 0, "No se generó error específico para nivel de significancia inválido"
    
    @given(st.integers().filter(lambda x: x < 2))
    def test_invalid_cv_folds_returns_error(self, invalid_folds):
        """
        **Propiedad: Número de folds de validación cruzada < 2 debe retornar error**
        
        **Validates: Requirements 9.4**
        """
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "training": {
                "cv_folds": invalid_folds
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"cv_folds inválido {invalid_folds} fue aceptado"
        
        # Verificar error específico
        cv_errors = [e for e in errors if "cv_folds" in e.path.lower()]
        assert len(cv_errors) > 0, "No se generó error específico para cv_folds inválido"
    
    @given(
        st.floats(min_value=0.5, max_value=0.9),
        st.floats(min_value=0.5, max_value=0.9)
    )
    def test_test_and_validation_size_sum_exceeding_one_returns_error(self, test_size, val_size):
        """
        **Propiedad: test_size + validation_size >= 1 debe retornar error**
        
        **Validates: Requirements 9.4**
        """
        assume(test_size + val_size >= 1)
        
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "training": {
                "test_size": test_size,
                "validation_size": val_size
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Suma de tamaños >= 1 fue aceptada: {test_size} + {val_size}"
        
        # Verificar error específico
        training_errors = [e for e in errors if "training" in e.path.lower()]
        assert len(training_errors) > 0, "No se generó error específico para suma de tamaños inválida"
    
    @given(st.lists(st.floats(), min_size=0, max_size=5).filter(lambda x: len(x) != 2))
    def test_invalid_optimal_range_format_returns_error(self, invalid_range):
        """
        **Propiedad: Rangos óptimos que no son listas de 2 elementos deben retornar error**
        
        **Validates: Requirements 9.4**
        """
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "recommendations": {
                "optimal_ranges": {
                    "study_hours": invalid_range
                }
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Rango inválido {invalid_range} fue aceptado"
        
        # Verificar error específico
        range_errors = [e for e in errors if "optimal_ranges" in e.path.lower()]
        assert len(range_errors) > 0, "No se generó error específico para formato de rango inválido"
    
    @given(
        st.floats(min_value=0.0, max_value=100.0),
        st.floats(min_value=0.0, max_value=100.0)
    )
    def test_invalid_optimal_range_order_returns_error(self, val1, val2):
        """
        **Propiedad: Rangos donde min >= max deben retornar error**
        
        **Validates: Requirements 9.4**
        """
        assume(val1 >= val2)
        
        config = {
            "project": {"name": "test"},
            "data": {"raw_path": "test.csv", "required_columns": ["col1"]},
            "recommendations": {
                "optimal_ranges": {
                    "study_hours": [val1, val2]
                }
            }
        }
        
        is_valid, errors = validate_config(config)
        
        assert not is_valid, f"Rango con min >= max fue aceptado: [{val1}, {val2}]"
        
        # Verificar error específico
        range_errors = [e for e in errors if "optimal_ranges" in e.path.lower()]
        assert len(range_errors) > 0, "No se generó error específico para orden de rango inválido"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
