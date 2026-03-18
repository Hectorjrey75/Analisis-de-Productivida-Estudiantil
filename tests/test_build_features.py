"""
Tests unitarios para el módulo src/features/build_features.py.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.features.build_features import encode_categorical, normalize_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'study_hours': [2.0, 4.0, 6.0, 8.0, 10.0],
        'sleep_hours': [5.0, 6.0, 7.0, 8.0, 9.0],
        'phone_usage': [1.0, 2.0, 3.0, 4.0, 5.0],
        'gender': ['M', 'F', 'M', 'F', 'M'],  # columna no numérica
    })


# ---------------------------------------------------------------------------
# normalize_features — método standard (z-score)
# ---------------------------------------------------------------------------

def test_standard_returns_dataframe_and_scaler(sample_df):
    result_df, scaler = normalize_features(sample_df, method='standard')
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)


def test_standard_numeric_columns_have_zero_mean(sample_df):
    result_df, _ = normalize_features(sample_df, method='standard')
    numeric_cols = sample_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        assert abs(result_df[col].mean()) < 1e-9, f"Media de '{col}' no es ~0"


def test_standard_numeric_columns_have_unit_std(sample_df):
    result_df, _ = normalize_features(sample_df, method='standard')
    numeric_cols = sample_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        assert abs(result_df[col].std(ddof=0) - 1.0) < 1e-9, f"Std de '{col}' no es ~1"


def test_standard_non_numeric_columns_unchanged(sample_df):
    result_df, _ = normalize_features(sample_df, method='standard')
    pd.testing.assert_series_equal(result_df['gender'], sample_df['gender'])


def test_standard_output_shape_matches_input(sample_df):
    result_df, _ = normalize_features(sample_df, method='standard')
    assert result_df.shape == sample_df.shape


# ---------------------------------------------------------------------------
# normalize_features — método minmax
# ---------------------------------------------------------------------------

def test_minmax_returns_dataframe_and_scaler(sample_df):
    result_df, scaler = normalize_features(sample_df, method='minmax')
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)


def test_minmax_numeric_columns_in_zero_one_range(sample_df):
    result_df, _ = normalize_features(sample_df, method='minmax')
    numeric_cols = sample_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        assert result_df[col].min() >= 0.0 - 1e-9, f"Min de '{col}' < 0"
        assert result_df[col].max() <= 1.0 + 1e-9, f"Max de '{col}' > 1"


def test_minmax_non_numeric_columns_unchanged(sample_df):
    result_df, _ = normalize_features(sample_df, method='minmax')
    pd.testing.assert_series_equal(result_df['gender'], sample_df['gender'])


def test_minmax_output_shape_matches_input(sample_df):
    result_df, _ = normalize_features(sample_df, method='minmax')
    assert result_df.shape == sample_df.shape


# ---------------------------------------------------------------------------
# Casos edge
# ---------------------------------------------------------------------------

def test_invalid_method_raises_value_error(sample_df):
    with pytest.raises(ValueError, match="no válido"):
        normalize_features(sample_df, method='invalid')


def test_dataframe_with_no_numeric_columns():
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
    result_df, scaler = normalize_features(df, method='standard')
    pd.testing.assert_frame_equal(result_df, df)


def test_scaler_is_fitted_and_can_transform(sample_df):
    """El scaler retornado debe estar ajustado y poder transformar nuevos datos."""
    _, scaler = normalize_features(sample_df, method='standard')
    numeric_cols = sample_df.select_dtypes(include='number').columns
    new_data = sample_df[numeric_cols].iloc[:2]
    transformed = scaler.transform(new_data)
    assert transformed.shape == new_data.shape


def test_original_dataframe_not_mutated(sample_df):
    """La función no debe modificar el DataFrame original."""
    original_values = sample_df.copy()
    normalize_features(sample_df, method='standard')
    pd.testing.assert_frame_equal(sample_df, original_values)


# ---------------------------------------------------------------------------
# encode_categorical — método onehot
# ---------------------------------------------------------------------------

@pytest.fixture
def cat_df():
    return pd.DataFrame({
        'study_hours': [2.0, 4.0, 6.0],
        'gender': ['M', 'F', 'M'],
        'level': ['low', 'high', 'medium'],
    })


def test_onehot_returns_dataframe_and_encoder(cat_df):
    from sklearn.preprocessing import OneHotEncoder
    result_df, encoder = encode_categorical(cat_df, method='onehot')
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(encoder, OneHotEncoder)


def test_onehot_drops_original_categorical_columns(cat_df):
    result_df, _ = encode_categorical(cat_df, method='onehot')
    assert 'gender' not in result_df.columns
    assert 'level' not in result_df.columns


def test_onehot_adds_new_columns(cat_df):
    result_df, _ = encode_categorical(cat_df, method='onehot')
    # Debe haber columnas con prefijo gender_ y level_
    assert any(c.startswith('gender_') for c in result_df.columns)
    assert any(c.startswith('level_') for c in result_df.columns)


def test_onehot_preserves_numeric_columns(cat_df):
    result_df, _ = encode_categorical(cat_df, method='onehot')
    assert 'study_hours' in result_df.columns
    pd.testing.assert_series_equal(
        result_df['study_hours'].reset_index(drop=True),
        cat_df['study_hours'].reset_index(drop=True),
    )


def test_onehot_encoder_is_fitted(cat_df):
    _, encoder = encode_categorical(cat_df, method='onehot')
    # Si está ajustado, categories_ debe existir
    assert hasattr(encoder, 'categories_')


def test_onehot_binary_values(cat_df):
    result_df, _ = encode_categorical(cat_df, method='onehot')
    ohe_cols = [c for c in result_df.columns if c.startswith('gender_') or c.startswith('level_')]
    for col in ohe_cols:
        assert set(result_df[col].unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# encode_categorical — método label
# ---------------------------------------------------------------------------

def test_label_returns_dataframe_and_dict(cat_df):
    result_df, encoder = encode_categorical(cat_df, method='label')
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(encoder, dict)


def test_label_keeps_categorical_columns_as_integers(cat_df):
    result_df, _ = encode_categorical(cat_df, method='label')
    assert result_df['gender'].dtype in (int, 'int32', 'int64', np.int32, np.int64)
    assert result_df['level'].dtype in (int, 'int32', 'int64', np.int32, np.int64)


def test_label_encoder_dict_has_entry_per_column(cat_df):
    _, encoder = encode_categorical(cat_df, method='label')
    assert 'gender' in encoder
    assert 'level' in encoder


def test_label_encoder_can_inverse_transform(cat_df):
    from sklearn.preprocessing import LabelEncoder
    result_df, encoder = encode_categorical(cat_df, method='label')
    for col in ['gender', 'level']:
        le: LabelEncoder = encoder[col]
        recovered = le.inverse_transform(result_df[col].values)
        assert list(recovered) == list(cat_df[col])


def test_label_preserves_numeric_columns(cat_df):
    result_df, _ = encode_categorical(cat_df, method='label')
    assert 'study_hours' in result_df.columns
    pd.testing.assert_series_equal(result_df['study_hours'], cat_df['study_hours'])


def test_label_shape_matches_input(cat_df):
    result_df, _ = encode_categorical(cat_df, method='label')
    assert result_df.shape == cat_df.shape


# ---------------------------------------------------------------------------
# encode_categorical — casos edge
# ---------------------------------------------------------------------------

def test_encode_invalid_method_raises_value_error(cat_df):
    with pytest.raises(ValueError, match="no válido"):
        encode_categorical(cat_df, method='invalid')


def test_encode_no_categorical_columns_returns_unchanged():
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    result_df, encoder = encode_categorical(df, method='onehot')
    pd.testing.assert_frame_equal(result_df, df)


def test_encode_no_categorical_columns_label_returns_empty_dict():
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    result_df, encoder = encode_categorical(df, method='label')
    assert encoder == {}
    pd.testing.assert_frame_equal(result_df, df)


def test_original_dataframe_not_mutated_by_encode(cat_df):
    original = cat_df.copy()
    encode_categorical(cat_df, method='label')
    pd.testing.assert_frame_equal(cat_df, original)


# ---------------------------------------------------------------------------
# handle_outliers
# ---------------------------------------------------------------------------

from src.features.build_features import handle_outliers


@pytest.fixture
def outlier_df():
    """DataFrame con un outlier claro en 'score'."""
    return pd.DataFrame({
        'score': [10.0, 11.0, 10.5, 9.5, 10.2, 100.0],   # 100 es outlier
        'hours': [5.0, 6.0, 5.5, 4.5, 5.2, 5.0],
        'label': ['a', 'b', 'a', 'b', 'a', 'b'],          # columna no numérica
    })


# --- estrategia clip ---

def test_clip_returns_dataframe(outlier_df):
    result = handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    assert isinstance(result, pd.DataFrame)


def test_clip_outlier_is_capped(outlier_df):
    result = handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    mean = outlier_df['score'].mean()
    std = outlier_df['score'].std()
    upper = mean + 2.0 * std
    assert result['score'].max() <= upper + 1e-9


def test_clip_non_outlier_values_unchanged(outlier_df):
    result = handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    # Los valores normales (no outliers) deben permanecer iguales
    normal_mask = outlier_df['score'] < 20
    pd.testing.assert_series_equal(
        result.loc[normal_mask, 'score'].reset_index(drop=True),
        outlier_df.loc[normal_mask, 'score'].reset_index(drop=True),
    )


def test_clip_non_numeric_columns_unchanged(outlier_df):
    result = handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    pd.testing.assert_series_equal(result['label'], outlier_df['label'])


def test_clip_shape_preserved(outlier_df):
    result = handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    assert result.shape == outlier_df.shape


def test_clip_does_not_mutate_original(outlier_df):
    original = outlier_df.copy()
    handle_outliers(outlier_df, strategy='clip', threshold=2.0)
    pd.testing.assert_frame_equal(outlier_df, original)


# --- estrategia remove ---

def test_remove_returns_dataframe(outlier_df):
    result = handle_outliers(outlier_df, strategy='remove', threshold=2.0)
    assert isinstance(result, pd.DataFrame)


def test_remove_drops_outlier_rows(outlier_df):
    result = handle_outliers(outlier_df, strategy='remove', threshold=2.0)
    # La fila con score=100 debe haber sido eliminada
    assert 100.0 not in result['score'].values


def test_remove_fewer_rows_than_original(outlier_df):
    result = handle_outliers(outlier_df, strategy='remove', threshold=2.0)
    assert len(result) < len(outlier_df)


def test_remove_non_numeric_columns_preserved(outlier_df):
    result = handle_outliers(outlier_df, strategy='remove', threshold=2.0)
    assert 'label' in result.columns


def test_remove_no_outliers_keeps_all_rows():
    df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = handle_outliers(df, strategy='remove', threshold=3.0)
    assert len(result) == len(df)


# --- estrategia winsorize ---

def test_winsorize_returns_dataframe(outlier_df):
    result = handle_outliers(outlier_df, strategy='winsorize', threshold=2.0)
    assert isinstance(result, pd.DataFrame)


def test_winsorize_shape_preserved(outlier_df):
    result = handle_outliers(outlier_df, strategy='winsorize', threshold=2.0)
    assert result.shape == outlier_df.shape


def test_winsorize_outlier_is_capped(outlier_df):
    result = handle_outliers(outlier_df, strategy='winsorize', threshold=2.0)
    # El valor extremo debe haber sido reducido
    assert result['score'].max() < outlier_df['score'].max()


def test_winsorize_non_numeric_columns_unchanged(outlier_df):
    result = handle_outliers(outlier_df, strategy='winsorize', threshold=2.0)
    pd.testing.assert_series_equal(result['label'], outlier_df['label'])


# --- casos edge ---

def test_handle_outliers_invalid_strategy_raises(outlier_df):
    with pytest.raises(ValueError, match="no válida"):
        handle_outliers(outlier_df, strategy='invalid')


def test_handle_outliers_no_numeric_columns_returns_unchanged():
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
    for strategy in ('clip', 'remove', 'winsorize'):
        result = handle_outliers(df, strategy=strategy)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# create_derived_features
# ---------------------------------------------------------------------------

from src.features.build_features import create_derived_features


@pytest.fixture
def full_df():
    """DataFrame con todas las columnas fuente para features derivadas."""
    return pd.DataFrame({
        'phone_usage_hours': [1.0, 2.0, 3.0],
        'social_media_hours': [0.5, 1.0, 1.5],
        'youtube_hours': [0.5, 1.0, 0.5],
        'gaming_hours': [0.0, 0.5, 1.0],
        'study_hours_per_day': [4.0, 6.0, 8.0],
        'sleep_hours': [7.0, 8.0, 6.0],
    })


def test_derived_returns_dataframe(full_df):
    result = create_derived_features(full_df)
    assert isinstance(result, pd.DataFrame)


def test_derived_preserves_original_columns(full_df):
    result = create_derived_features(full_df)
    for col in full_df.columns:
        assert col in result.columns


def test_derived_does_not_mutate_original(full_df):
    original = full_df.copy()
    create_derived_features(full_df)
    pd.testing.assert_frame_equal(full_df, original)


def test_total_screen_time_correct(full_df):
    result = create_derived_features(full_df)
    expected = full_df[['phone_usage_hours', 'social_media_hours', 'youtube_hours', 'gaming_hours']].sum(axis=1)
    pd.testing.assert_series_equal(result['total_screen_time'], expected, check_names=False)


def test_study_sleep_ratio_correct(full_df):
    result = create_derived_features(full_df)
    expected = full_df['study_hours_per_day'] / full_df['sleep_hours']
    pd.testing.assert_series_equal(result['study_sleep_ratio'], expected, check_names=False)


def test_digital_distraction_index_correct(full_df):
    result = create_derived_features(full_df)
    expected = (full_df['social_media_hours'] + full_df['gaming_hours']) / full_df['study_hours_per_day']
    pd.testing.assert_series_equal(result['digital_distraction_index'], expected, check_names=False)


def test_screen_to_study_ratio_correct(full_df):
    result = create_derived_features(full_df)
    expected = result['total_screen_time'] / full_df['study_hours_per_day']
    pd.testing.assert_series_equal(result['screen_to_study_ratio'], expected, check_names=False)


def test_derived_skips_missing_screen_cols():
    """Si faltan columnas de pantalla, no se crea total_screen_time."""
    df = pd.DataFrame({
        'study_hours_per_day': [4.0, 6.0],
        'sleep_hours': [7.0, 8.0],
    })
    result = create_derived_features(df)
    assert 'total_screen_time' not in result.columns
    assert 'screen_to_study_ratio' not in result.columns


def test_derived_skips_study_sleep_ratio_when_missing():
    """Si falta sleep_hours, no se crea study_sleep_ratio."""
    df = pd.DataFrame({
        'phone_usage_hours': [1.0],
        'social_media_hours': [0.5],
        'youtube_hours': [0.5],
        'gaming_hours': [0.0],
        'study_hours_per_day': [4.0],
    })
    result = create_derived_features(df)
    assert 'study_sleep_ratio' not in result.columns


def test_derived_handles_zero_sleep_hours():
    """División por cero en sleep_hours debe producir 0 (fillna)."""
    df = pd.DataFrame({
        'study_hours_per_day': [4.0],
        'sleep_hours': [0.0],
    })
    result = create_derived_features(df)
    assert result['study_sleep_ratio'].iloc[0] == 0.0


def test_derived_handles_zero_study_hours():
    """División por cero en study_hours_per_day debe producir 0 (fillna)."""
    df = pd.DataFrame({
        'social_media_hours': [1.0],
        'gaming_hours': [0.5],
        'study_hours_per_day': [0.0],
        'sleep_hours': [7.0],
    })
    result = create_derived_features(df)
    assert result['digital_distraction_index'].iloc[0] == 0.0
