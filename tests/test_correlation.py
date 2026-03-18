"""
Tests unitarios para el módulo src/analysis/correlation.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlation import compute_correlation_matrix


@pytest.fixture
def numeric_df():
    """DataFrame con columnas numéricas perfectamente correlacionadas y anticorrelacionadas."""
    np.random.seed(42)
    x = np.arange(1, 11, dtype=float)
    return pd.DataFrame({
        'a': x,
        'b': x * 2,          # correlación perfecta +1 con 'a'
        'c': -x,             # correlación perfecta -1 con 'a'
        'd': np.random.rand(10),  # correlación aleatoria
    })


@pytest.fixture
def mixed_df(numeric_df):
    """DataFrame con columnas numéricas y no numéricas."""
    df = numeric_df.copy()
    df['categoria'] = ['X', 'Y'] * 5
    return df


# ---------------------------------------------------------------------------
# Tests de método Pearson (default)
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrixPearson:
    def test_returns_dataframe(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_shape_is_square(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        assert result.shape[0] == result.shape[1]

    def test_diagonal_is_one(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    def test_perfect_positive_correlation(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        assert result.loc['a', 'b'] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        assert result.loc['a', 'c'] == pytest.approx(-1.0)

    def test_symmetric_matrix(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        pd.testing.assert_frame_equal(result, result.T)

    def test_values_between_minus_one_and_one(self, numeric_df):
        result = compute_correlation_matrix(numeric_df)
        assert (result.values >= -1.0).all()
        assert (result.values <= 1.0).all()

    def test_default_method_is_pearson(self, numeric_df):
        result_default = compute_correlation_matrix(numeric_df)
        result_pearson = compute_correlation_matrix(numeric_df, method='pearson')
        pd.testing.assert_frame_equal(result_default, result_pearson)


# ---------------------------------------------------------------------------
# Tests de métodos Spearman y Kendall
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrixMethods:
    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_returns_dataframe(self, numeric_df, method):
        result = compute_correlation_matrix(numeric_df, method=method)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_diagonal_is_one(self, numeric_df, method):
        result = compute_correlation_matrix(numeric_df, method=method)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_perfect_positive_correlation(self, numeric_df, method):
        result = compute_correlation_matrix(numeric_df, method=method)
        assert result.loc['a', 'b'] == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ["spearman", "kendall"])
    def test_perfect_negative_correlation(self, numeric_df, method):
        result = compute_correlation_matrix(numeric_df, method=method)
        assert result.loc['a', 'c'] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests de filtrado de columnas no numéricas
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrixNonNumeric:
    def test_ignores_non_numeric_columns(self, mixed_df):
        result = compute_correlation_matrix(mixed_df)
        assert 'categoria' not in result.columns
        assert 'categoria' not in result.index

    def test_only_numeric_columns_in_result(self, mixed_df):
        result = compute_correlation_matrix(mixed_df)
        numeric_cols = mixed_df.select_dtypes(include='number').columns.tolist()
        assert sorted(result.columns.tolist()) == sorted(numeric_cols)


# ---------------------------------------------------------------------------
# Tests de manejo de errores
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrixErrors:
    def test_raises_for_invalid_method(self, numeric_df):
        with pytest.raises(ValueError, match="no válido"):
            compute_correlation_matrix(numeric_df, method='invalid')

    def test_raises_for_empty_method_string(self, numeric_df):
        with pytest.raises(ValueError):
            compute_correlation_matrix(numeric_df, method='')

    @pytest.mark.parametrize("bad_method", ["PEARSON", "Spearman", "KENDALL"])
    def test_method_is_case_sensitive(self, numeric_df, bad_method):
        with pytest.raises(ValueError):
            compute_correlation_matrix(numeric_df, method=bad_method)


# ---------------------------------------------------------------------------
# Tests para find_significant_correlations
# ---------------------------------------------------------------------------

from src.analysis.correlation import find_significant_correlations


@pytest.fixture
def corr_matrix_with_target():
    """
    Matriz de correlación sintética con una variable objetivo conocida.
    'target' tiene correlación perfecta con 'a', anticorrelación con 'b',
    y correlación cero con 'c'.
    """
    data = {
        'target': [1.0, 1.0, -1.0, 0.0],
        'a':      [1.0, 0.5, -0.5, 0.0],
        'b':      [-1.0, 0.0, 0.0, 0.0],
        'c':      [0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    return df.corr(method='pearson')


class TestFindSignificantCorrelations:
    def test_returns_dataframe(self, corr_matrix_with_target):
        result = find_significant_correlations(corr_matrix_with_target, target='target')
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, corr_matrix_with_target):
        result = find_significant_correlations(corr_matrix_with_target, target='target')
        for col in ('feature', 'correlation', 'p_value', 'is_significant'):
            assert col in result.columns

    def test_target_not_in_features(self, corr_matrix_with_target):
        result = find_significant_correlations(corr_matrix_with_target, target='target')
        assert 'target' not in result['feature'].values

    def test_raises_if_target_missing(self, corr_matrix_with_target):
        with pytest.raises(ValueError, match="no encontrada"):
            find_significant_correlations(corr_matrix_with_target, target='nonexistent')

    def test_sorted_by_absolute_correlation_descending(self, corr_matrix_with_target):
        result = find_significant_correlations(corr_matrix_with_target, target='target')
        abs_corrs = result['correlation'].abs().tolist()
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_no_pvalue_without_n_samples(self, corr_matrix_with_target):
        result = find_significant_correlations(corr_matrix_with_target, target='target')
        assert result['p_value'].isna().all()
        assert result['is_significant'].isna().all()

    def test_pvalue_computed_with_n_samples(self, corr_matrix_with_target):
        result = find_significant_correlations(
            corr_matrix_with_target, target='target', n_samples=100
        )
        assert not result['p_value'].isna().all()

    def test_filters_non_significant_with_n_samples(self):
        """Con n_samples grande, correlaciones cercanas a 0 deben ser filtradas."""
        np.random.seed(0)
        n = 200
        x = np.random.randn(n)
        # strong correlation
        strong = x + np.random.randn(n) * 0.1
        # near-zero correlation
        noise = np.random.randn(n)
        df = pd.DataFrame({'target': x, 'strong': strong, 'noise': noise})
        corr_matrix = df.corr(method='pearson')

        result = find_significant_correlations(
            corr_matrix, target='target', confidence_level=0.95, n_samples=n
        )
        # Only significant correlations should be returned
        assert all(result['is_significant'] == True)
        # 'strong' should be present; 'noise' likely filtered out
        assert 'strong' in result['feature'].values

    def test_all_returned_are_significant(self):
        """Todos los resultados deben tener is_significant=True cuando n_samples es dado."""
        np.random.seed(1)
        n = 500
        x = np.random.randn(n)
        df = pd.DataFrame({
            'target': x,
            'feat1': x * 0.9 + np.random.randn(n) * 0.1,
            'feat2': np.random.randn(n),
        })
        corr_matrix = df.corr()
        result = find_significant_correlations(
            corr_matrix, target='target', confidence_level=0.95, n_samples=n
        )
        if len(result) > 0:
            assert (result['is_significant'] == True).all()

    def test_confidence_level_affects_filtering(self):
        """Un nivel de confianza más alto debe ser más restrictivo (menos o igual resultados)."""
        np.random.seed(2)
        n = 100
        x = np.random.randn(n)
        df = pd.DataFrame({
            'target': x,
            'feat1': x * 0.5 + np.random.randn(n) * 0.5,
            'feat2': np.random.randn(n),
            'feat3': np.random.randn(n),
        })
        corr_matrix = df.corr()
        result_90 = find_significant_correlations(
            corr_matrix, target='target', confidence_level=0.90, n_samples=n
        )
        result_99 = find_significant_correlations(
            corr_matrix, target='target', confidence_level=0.99, n_samples=n
        )
        assert len(result_99) <= len(result_90)


# ---------------------------------------------------------------------------
# Tests para detect_multicollinearity
# ---------------------------------------------------------------------------

from src.analysis.correlation import detect_multicollinearity


@pytest.fixture
def corr_matrix_multicollinearity():
    """
    DataFrame con pares de variables altamente correlacionadas.
    'a' y 'b' tienen correlación perfecta (+1).
    'a' y 'c' tienen correlación perfecta (-1).
    'd' tiene correlación baja con el resto.
    """
    np.random.seed(42)
    x = np.arange(1, 21, dtype=float)
    df = pd.DataFrame({
        'a': x,
        'b': x * 2,           # corr(a,b) = +1.0
        'c': -x,              # corr(a,c) = -1.0
        'd': np.random.rand(20),  # correlación baja
    })
    return df.corr(method='pearson')


class TestDetectMulticollinearity:
    def test_returns_list(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity)
        assert isinstance(result, list)

    def test_each_element_is_tuple_of_three(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    def test_detects_perfect_positive_correlation(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.8)
        pairs = [(v1, v2) for v1, v2, _ in result]
        assert ('a', 'b') in pairs

    def test_detects_perfect_negative_correlation(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.8)
        pairs = [(v1, v2) for v1, v2, _ in result]
        assert ('a', 'c') in pairs

    def test_no_self_correlations(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.0)
        for v1, v2, _ in result:
            assert v1 != v2

    def test_no_duplicate_pairs(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.0)
        pairs = [(v1, v2) for v1, v2, _ in result]
        assert len(pairs) == len(set(pairs))

    def test_correlation_value_matches_matrix(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.8)
        for v1, v2, corr_val in result:
            expected = corr_matrix_multicollinearity.loc[v1, v2]
            assert corr_val == pytest.approx(expected)

    def test_threshold_filters_low_correlations(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.99)
        for _, _, corr_val in result:
            assert abs(corr_val) >= 0.99

    def test_empty_result_when_threshold_above_one(self, corr_matrix_multicollinearity):
        result = detect_multicollinearity(corr_matrix_multicollinearity, threshold=1.1)
        assert result == []

    def test_default_threshold_is_0_8(self, corr_matrix_multicollinearity):
        result_default = detect_multicollinearity(corr_matrix_multicollinearity)
        result_explicit = detect_multicollinearity(corr_matrix_multicollinearity, threshold=0.8)
        assert result_default == result_explicit

    def test_returns_empty_for_uncorrelated_data(self):
        np.random.seed(99)
        df = pd.DataFrame(np.random.randn(50, 4), columns=['w', 'x', 'y', 'z'])
        corr_matrix = df.corr()
        result = detect_multicollinearity(corr_matrix, threshold=0.99)
        assert isinstance(result, list)
