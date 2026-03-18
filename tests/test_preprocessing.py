"""
Tests unitarios y de propiedad para DataPreprocessor.

**Validates: Requisito 10.6**
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import column, data_frames

from src.data.preprocessing import DataPreprocessor


# ---------------------------------------------------------------------------
# Configuraciones de prueba
# ---------------------------------------------------------------------------

CONFIG_STANDARD = {
    "preprocessing": {
        "normalization": {"method": "standard"},
        "encoding": {"method": "label"},
    }
}

CONFIG_MINMAX = {
    "preprocessing": {
        "normalization": {"method": "minmax"},
        "encoding": {"method": "label"},
    }
}


# ---------------------------------------------------------------------------
# Estrategia Hypothesis: DataFrames numéricos válidos
# ---------------------------------------------------------------------------

def numeric_dataframe_strategy(min_rows=2, max_rows=50, min_cols=1, max_cols=5):
    """Genera DataFrames con columnas numéricas finitas y sin NaN."""
    col_names = [f"col_{i}" for i in range(max_cols)]

    @st.composite
    def _strategy(draw):
        n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
        n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
        selected_cols = col_names[:n_cols]

        data = {}
        for col in selected_cols:
            values = draw(
                st.lists(
                    st.floats(
                        min_value=-1e6,
                        max_value=1e6,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
            data[col] = values

        return pd.DataFrame(data)

    return _strategy()


# ---------------------------------------------------------------------------
# Tests unitarios
# ---------------------------------------------------------------------------

class TestDataPreprocessorUnit:
    """Tests unitarios para DataPreprocessor."""

    def _sample_df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_fit_sets_scaler_transformer(self):
        """fit() debe registrar un scaler en self.transformers."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        df = self._sample_df()
        preprocessor.fit(df)

        assert "scaler" in preprocessor.transformers
        assert preprocessor._numeric_cols == ["a", "b"]

    def test_fit_minmax_sets_minmax_scaler(self):
        """fit() con método minmax debe usar MinMaxScaler."""
        from sklearn.preprocessing import MinMaxScaler
        preprocessor = DataPreprocessor(CONFIG_MINMAX)
        preprocessor.fit(self._sample_df())

        assert isinstance(preprocessor.transformers["scaler"], MinMaxScaler)

    def test_fit_standard_sets_standard_scaler(self):
        """fit() con método standard debe usar StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        preprocessor.fit(self._sample_df())

        assert isinstance(preprocessor.transformers["scaler"], StandardScaler)

    def test_transform_changes_numeric_values(self):
        """transform() debe modificar los valores numéricos."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        df = self._sample_df()
        preprocessor.fit(df)
        transformed = preprocessor.transform(df)

        # Los valores normalizados no deben ser iguales a los originales
        assert not np.allclose(transformed["a"].values, df["a"].values)
        assert not np.allclose(transformed["b"].values, df["b"].values)

    def test_inverse_transform_restores_original_values(self):
        """inverse_transform() debe restaurar los valores originales."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        df = self._sample_df()
        preprocessor.fit(df)
        transformed = preprocessor.transform(df)
        restored = preprocessor.inverse_transform(transformed)

        np.testing.assert_allclose(
            restored["a"].values, df["a"].values, atol=1e-6
        )
        np.testing.assert_allclose(
            restored["b"].values, df["b"].values, atol=1e-6
        )

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        """fit_transform() debe producir el mismo resultado que fit() + transform()."""
        df = self._sample_df()

        p1 = DataPreprocessor(CONFIG_STANDARD)
        result_combined = p1.fit_transform(df)

        p2 = DataPreprocessor(CONFIG_STANDARD)
        p2.fit(df)
        result_separate = p2.transform(df)

        pd.testing.assert_frame_equal(result_combined, result_separate)

    def test_transform_without_fit_raises_error(self):
        """transform() sin fit() previo debe lanzar RuntimeError."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        with pytest.raises(RuntimeError):
            preprocessor.transform(self._sample_df())

    def test_inverse_transform_without_fit_raises_error(self):
        """inverse_transform() sin fit() previo debe lanzar RuntimeError."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        with pytest.raises(RuntimeError):
            preprocessor.inverse_transform(self._sample_df())


# ---------------------------------------------------------------------------
# Tests de propiedad (Hypothesis)
# ---------------------------------------------------------------------------

class TestPreprocessorReversibilityProperty:
    """
    Tests de propiedad: fit_transform() seguido de inverse_transform()
    debe producir valores equivalentes dentro de tolerancia 1e-6.

    **Validates: Requisito 10.6**
    """

    @given(numeric_dataframe_strategy())
    @settings(max_examples=50)
    def test_standard_normalization_is_reversible(self, df):
        """
        **Propiedad: Transformación estándar es reversible**

        Para cualquier DataFrame numérico válido, aplicar fit_transform()
        seguido de inverse_transform() debe producir valores dentro de
        tolerancia 1e-6 respecto a los originales.

        **Validates: Requisito 10.6**
        """
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        transformed = preprocessor.fit_transform(df)
        restored = preprocessor.inverse_transform(transformed)

        for col in df.columns:
            np.testing.assert_allclose(
                restored[col].values,
                df[col].values,
                atol=1e-6,
                err_msg=f"Columna '{col}' no fue restaurada correctamente (standard)",
            )

    @given(numeric_dataframe_strategy())
    @settings(max_examples=50)
    def test_minmax_normalization_is_reversible(self, df):
        """
        **Propiedad: Transformación minmax es reversible**

        Para cualquier DataFrame numérico válido, aplicar fit_transform()
        seguido de inverse_transform() debe producir valores dentro de
        tolerancia 1e-6 respecto a los originales.

        **Validates: Requisito 10.6**
        """
        preprocessor = DataPreprocessor(CONFIG_MINMAX)
        transformed = preprocessor.fit_transform(df)
        restored = preprocessor.inverse_transform(transformed)

        for col in df.columns:
            np.testing.assert_allclose(
                restored[col].values,
                df[col].values,
                atol=1e-6,
                err_msg=f"Columna '{col}' no fue restaurada correctamente (minmax)",
            )


# ---------------------------------------------------------------------------
# Tests de persistencia de transformadores (tarea 3.4)
# ---------------------------------------------------------------------------

class TestSaveLoadTransformers:
    """Tests para save_transformers() y load_transformers()."""

    def _sample_df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_save_and_load_transformers_roundtrip(self, tmp_path):
        """Guardar y cargar transformadores debe permitir transform() sin fit()."""
        path = str(tmp_path / "transformers.pkl")

        # Ajustar y guardar
        p1 = DataPreprocessor(CONFIG_STANDARD)
        df = self._sample_df()
        p1.fit(df)
        p1.save_transformers(path)

        # Cargar en un nuevo preprocesador y transformar
        p2 = DataPreprocessor(CONFIG_STANDARD)
        p2.load_transformers(path)
        result = p2.transform(df)

        # Debe producir el mismo resultado que el original
        expected = p1.transform(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_save_without_fit_raises_error(self, tmp_path):
        """save_transformers() sin fit() previo debe lanzar RuntimeError."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        with pytest.raises(RuntimeError):
            preprocessor.save_transformers(str(tmp_path / "transformers.pkl"))

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """load_transformers() con archivo inexistente debe lanzar FileNotFoundError."""
        preprocessor = DataPreprocessor(CONFIG_STANDARD)
        with pytest.raises(FileNotFoundError):
            preprocessor.load_transformers(str(tmp_path / "no_existe.pkl"))

    def test_loaded_transformers_support_inverse_transform(self, tmp_path):
        """Después de load_transformers(), inverse_transform() debe funcionar correctamente."""
        path = str(tmp_path / "transformers.pkl")
        df = self._sample_df()

        p1 = DataPreprocessor(CONFIG_STANDARD)
        p1.fit(df)
        p1.save_transformers(path)

        p2 = DataPreprocessor(CONFIG_STANDARD)
        p2.load_transformers(path)
        transformed = p2.transform(df)
        restored = p2.inverse_transform(transformed)

        np.testing.assert_allclose(restored["a"].values, df["a"].values, atol=1e-6)
        np.testing.assert_allclose(restored["b"].values, df["b"].values, atol=1e-6)

    def test_save_load_preserves_minmax_scaler(self, tmp_path):
        """save/load debe preservar el tipo de scaler (MinMaxScaler)."""
        from sklearn.preprocessing import MinMaxScaler
        path = str(tmp_path / "transformers.pkl")
        df = self._sample_df()

        p1 = DataPreprocessor(CONFIG_MINMAX)
        p1.fit(df)
        p1.save_transformers(path)

        p2 = DataPreprocessor(CONFIG_MINMAX)
        p2.load_transformers(path)

        assert isinstance(p2.transformers["scaler"], MinMaxScaler)
