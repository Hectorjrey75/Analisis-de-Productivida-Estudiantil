"""Tests para src/data/make_dataset.py — tareas 2.1 y 2.2."""

import pytest
import pandas as pd

from src.data.make_dataset import load_raw_data, validate_schema


# Config mínima que refleja config.yaml
MINIMAL_CONFIG = {
    "data": {
        "required_columns": ["student_id", "productivity_score"]
    }
}

EMPTY_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_csv(tmp_path):
    """CSV válido con columnas requeridas."""
    path = tmp_path / "students.csv"
    df = pd.DataFrame({
        "student_id": [1, 2, 3],
        "productivity_score": [0.8, 0.6, 0.9],
        "final_grade": [85.0, 70.0, 92.0],
    })
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def missing_column_csv(tmp_path):
    """CSV que no contiene todas las columnas requeridas."""
    path = tmp_path / "incomplete.csv"
    df = pd.DataFrame({"student_id": [1, 2]})
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def empty_csv(tmp_path):
    """CSV con cabecera pero sin filas."""
    path = tmp_path / "empty.csv"
    path.write_text("student_id,productivity_score\n")
    return str(path)


@pytest.fixture()
def invalid_file(tmp_path):
    """Archivo con contenido no-CSV."""
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\x00\x01\x02\x03")
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadRawDataSuccess:
    def test_returns_dataframe(self, valid_csv):
        df = load_raw_data(valid_csv, MINIMAL_CONFIG)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, valid_csv):
        df = load_raw_data(valid_csv, MINIMAL_CONFIG)
        assert len(df) == 3

    def test_columns_preserved(self, valid_csv):
        df = load_raw_data(valid_csv, MINIMAL_CONFIG)
        assert "student_id" in df.columns
        assert "productivity_score" in df.columns

    def test_no_required_columns_in_config(self, valid_csv):
        """Sin required_columns en config no debe lanzar error."""
        df = load_raw_data(valid_csv, EMPTY_CONFIG)
        assert isinstance(df, pd.DataFrame)


class TestLoadRawDataErrors:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(str(tmp_path / "nonexistent.csv"), MINIMAL_CONFIG)

    def test_missing_required_columns_raises(self, missing_column_csv):
        with pytest.raises(ValueError, match="Columnas requeridas faltantes"):
            load_raw_data(missing_column_csv, MINIMAL_CONFIG)

    def test_empty_file_raises(self, empty_csv):
        with pytest.raises(ValueError):
            load_raw_data(empty_csv, MINIMAL_CONFIG)


class TestValidateSchema:
    """Tests para validate_schema() — tarea 2.2 / Requisito 1.4."""

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "student_id": [1, 2],
            "productivity_score": [0.8, 0.6],
            "final_grade": [85.0, 70.0],
        })

    def test_all_columns_present_is_valid(self, sample_df):
        result = validate_schema(sample_df, ["student_id", "productivity_score"])
        assert result.is_valid is True
        assert result.errors == []

    def test_missing_column_is_invalid(self, sample_df):
        result = validate_schema(sample_df, ["student_id", "nonexistent_col"])
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "nonexistent_col" in result.errors[0]

    def test_multiple_missing_columns_reported(self, sample_df):
        result = validate_schema(sample_df, ["col_a", "col_b"])
        assert result.is_valid is False
        assert "col_a" in result.errors[0]
        assert "col_b" in result.errors[0]

    def test_empty_required_columns_is_valid(self, sample_df):
        result = validate_schema(sample_df, [])
        assert result.is_valid is True

    def test_quality_metrics_contain_missing_columns(self, sample_df):
        result = validate_schema(sample_df, ["student_id", "missing_col"])
        assert "missing_columns" in result.quality_metrics
        assert "missing_col" in result.quality_metrics["missing_columns"]

    def test_quality_metrics_total_columns(self, sample_df):
        result = validate_schema(sample_df, ["student_id"])
        assert result.quality_metrics["total_columns"] == len(sample_df.columns)


from src.data.make_dataset import compute_data_quality_stats


class TestComputeDataQualityStats:
    """Tests para compute_data_quality_stats() — tarea 2.3 / Requisito 1.5."""

    @pytest.fixture()
    def clean_df(self):
        return pd.DataFrame({
            "student_id": [1, 2, 3],
            "productivity_score": [0.8, 0.6, 0.9],
            "final_grade": [85.0, 70.0, 92.0],
        })

    @pytest.fixture()
    def df_with_nulls(self):
        return pd.DataFrame({
            "student_id": [1, 2, 3, 4],
            "productivity_score": [0.8, None, 0.9, None],
            "final_grade": [85.0, 70.0, None, 92.0],
        })

    @pytest.fixture()
    def df_with_duplicates(self):
        return pd.DataFrame({
            "student_id": [1, 2, 1],
            "productivity_score": [0.8, 0.6, 0.8],
        })

    def test_returns_dict(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        assert isinstance(result, dict)

    def test_required_keys_present(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        for key in ("null_percentages", "duplicate_rows", "duplicate_percentage",
                    "descriptive_stats", "total_rows", "total_columns"):
            assert key in result

    def test_total_rows_and_columns(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        assert result["total_rows"] == 3
        assert result["total_columns"] == 3

    def test_no_nulls_in_clean_df(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        assert all(v == 0.0 for v in result["null_percentages"].values())

    def test_null_percentages_correct(self, df_with_nulls):
        result = compute_data_quality_stats(df_with_nulls)
        # productivity_score: 2 nulos de 4 → 50%
        assert result["null_percentages"]["productivity_score"] == 50.0
        # final_grade: 1 nulo de 4 → 25%
        assert result["null_percentages"]["final_grade"] == 25.0

    def test_no_duplicates_in_clean_df(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        assert result["duplicate_rows"] == 0
        assert result["duplicate_percentage"] == 0.0

    def test_duplicate_rows_detected(self, df_with_duplicates):
        result = compute_data_quality_stats(df_with_duplicates)
        assert result["duplicate_rows"] == 1
        assert result["duplicate_percentage"] > 0.0

    def test_descriptive_stats_keys(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        for col_stats in result["descriptive_stats"].values():
            assert "mean" in col_stats
            assert "std" in col_stats
            assert "min" in col_stats
            assert "max" in col_stats

    def test_descriptive_stats_values(self, clean_df):
        result = compute_data_quality_stats(clean_df)
        stats = result["descriptive_stats"]["productivity_score"]
        assert stats["min"] == pytest.approx(0.6)
        assert stats["max"] == pytest.approx(0.9)
        assert stats["mean"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)

    def test_non_numeric_columns_excluded_from_descriptive_stats(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "score": [0.8, 0.6],
        })
        result = compute_data_quality_stats(df)
        assert "name" not in result["descriptive_stats"]
        assert "score" in result["descriptive_stats"]


from src.data.make_dataset import impute_missing_values, apply_imputation


class TestImputeMissingValues:
    """Tests para impute_missing_values() — tarea 2.4 / Requisito 1.2."""

    @pytest.fixture()
    def df_with_nulls(self):
        return pd.DataFrame({
            "score": [1.0, None, 3.0, None],
            "grade": [10.0, 20.0, None, 40.0],
            "label": ["a", None, "c", None],
        })

    def test_mean_fills_numeric_nulls(self, df_with_nulls):
        result = impute_missing_values(df_with_nulls, strategy='mean')
        assert result["score"].isna().sum() == 0
        assert result["grade"].isna().sum() == 0
        # mean of [1, 3] = 2.0
        assert result["score"].iloc[1] == pytest.approx(2.0)

    def test_median_fills_numeric_nulls(self, df_with_nulls):
        result = impute_missing_values(df_with_nulls, strategy='median')
        assert result["score"].isna().sum() == 0
        assert result["grade"].isna().sum() == 0
        # median of [1, 3] = 2.0
        assert result["score"].iloc[1] == pytest.approx(2.0)

    def test_mode_fills_all_columns(self, df_with_nulls):
        result = impute_missing_values(df_with_nulls, strategy='mode')
        assert result["score"].isna().sum() == 0
        assert result["label"].isna().sum() == 0

    def test_forward_fill_propagates_values(self):
        df = pd.DataFrame({"x": [1.0, None, None, 4.0]})
        result = impute_missing_values(df, strategy='forward_fill')
        assert result["x"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_drop_removes_rows_with_nulls(self, df_with_nulls):
        result = impute_missing_values(df_with_nulls, strategy='drop')
        assert result.isna().sum().sum() == 0
        assert len(result) < len(df_with_nulls)

    def test_invalid_strategy_raises(self, df_with_nulls):
        with pytest.raises(ValueError, match="Estrategia de imputación inválida"):
            impute_missing_values(df_with_nulls, strategy='unknown')

    def test_no_nulls_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = impute_missing_values(df, strategy='median')
        pd.testing.assert_frame_equal(result, df)

    def test_original_df_not_mutated(self, df_with_nulls):
        original_nulls = df_with_nulls.isna().sum().sum()
        impute_missing_values(df_with_nulls, strategy='mean')
        assert df_with_nulls.isna().sum().sum() == original_nulls


class TestApplyImputation:
    """Tests para apply_imputation() — tarea 2.4 / Requisito 1.2."""

    @pytest.fixture()
    def df_with_nulls(self):
        return pd.DataFrame({"score": [1.0, None, 3.0]})

    def test_reads_strategy_from_config(self, df_with_nulls):
        config = {"data": {"imputation": {"strategy": "mean"}}}
        result = apply_imputation(df_with_nulls, config)
        assert result["score"].isna().sum() == 0

    def test_defaults_to_median_when_missing(self, df_with_nulls):
        result = apply_imputation(df_with_nulls, {})
        assert result["score"].isna().sum() == 0

    def test_nested_config_key_used(self, df_with_nulls):
        config = {"data": {"imputation": {"strategy": "drop"}}}
        result = apply_imputation(df_with_nulls, config)
        assert len(result) == 2  # one row dropped
