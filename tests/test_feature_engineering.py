"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import ALL_FEATURES, load_all_patients
from src.data.feature_engineering import (
    INPUT_FEATURES,
    N_TOTAL_FEATURES,
    forward_fill_patient,
    create_missingness_features,
    create_time_since_features,
    impute_remaining,
    normalize_features,
    preprocess_patient,
)

DATA_DIR = "data/raw/training_setA"


def make_synthetic_patient(n_hours=5):
    """Create a synthetic patient DataFrame with known missing patterns."""
    data = {col: [np.nan] * n_hours for col in ALL_FEATURES}

    # Vitals with known patterns
    data["HR"] = [80.0, np.nan, np.nan, 90.0, 85.0]
    data["O2Sat"] = [95.0, 96.0, np.nan, 94.0, np.nan]
    data["Temp"] = [np.nan, np.nan, 37.0, np.nan, np.nan]
    data["Resp"] = [18.0, 20.0, 22.0, 19.0, 21.0]
    data["SBP"] = [120.0, 118.0, 115.0, 110.0, 105.0]
    data["MAP"] = [80.0, 78.0, 65.0, 70.0, 72.0]
    data["DBP"] = [70.0, 68.0, 55.0, 60.0, 62.0]

    # Lab values
    data["FiO2"] = [np.nan, np.nan, 0.4, np.nan, np.nan]
    data["Creatinine"] = [np.nan, 1.5, np.nan, np.nan, np.nan]
    data["Bilirubin_total"] = [np.nan, np.nan, np.nan, 2.0, np.nan]
    data["Platelets"] = [200.0, np.nan, np.nan, 100.0, np.nan]
    data["BUN"] = [np.nan, 30.0, np.nan, np.nan, np.nan]

    # Demographics (static)
    data["Age"] = [65.0] * n_hours
    data["Gender"] = [1.0] * n_hours
    data["Unit1"] = [np.nan] * n_hours
    data["Unit2"] = [1.0] * n_hours
    data["HospAdmTime"] = [-2.5] * n_hours
    data["ICULOS"] = list(range(1, n_hours + 1))

    # Target
    data["SepsisLabel"] = [0, 0, 0, 0, 1]
    return pd.DataFrame(data)


class TestForwardFill:
    def test_fills_within_sequence(self):
        df = make_synthetic_patient()
        filled = forward_fill_patient(df)
        assert filled["HR"].tolist() == [80.0, 80.0, 80.0, 90.0, 85.0]

    def test_preserves_leading_nans(self):
        df = make_synthetic_patient()
        filled = forward_fill_patient(df)
        assert pd.isna(filled["Temp"].iloc[0])
        assert pd.isna(filled["Temp"].iloc[1])
        assert filled["Temp"].iloc[2] == 37.0
        assert filled["Temp"].iloc[3] == 37.0
        assert filled["Temp"].iloc[4] == 37.0

    def test_fully_observed_unchanged(self):
        df = make_synthetic_patient()
        filled = forward_fill_patient(df)
        assert filled["Resp"].tolist() == [18.0, 20.0, 22.0, 19.0, 21.0]


class TestMissingnessFeatures:
    def test_mask_values(self):
        df = make_synthetic_patient()
        mask = create_missingness_features(df, ["HR", "Temp", "Resp"])
        assert mask["HR_missing"].tolist() == [0, 1, 1, 0, 0]
        assert mask["Temp_missing"].tolist() == [1, 1, 0, 1, 1]
        assert mask["Resp_missing"].tolist() == [0, 0, 0, 0, 0]

    def test_column_count(self):
        df = make_synthetic_patient()
        mask = create_missingness_features(df, INPUT_FEATURES)
        assert mask.shape[1] == 40


class TestTimeSinceFeatures:
    def test_observed_at_start(self):
        df = make_synthetic_patient()
        ts = create_time_since_features(df, ["HR"])
        np.testing.assert_array_equal(ts["HR_time_since"].values, [0, 1, 2, 0, 0])

    def test_missing_at_start(self):
        df = make_synthetic_patient()
        ts = create_time_since_features(df, ["Temp"])
        np.testing.assert_array_equal(ts["Temp_time_since"].values, [1, 2, 0, 1, 2])

    def test_fully_observed(self):
        df = make_synthetic_patient()
        ts = create_time_since_features(df, ["Resp"])
        np.testing.assert_array_equal(ts["Resp_time_since"].values, [0, 0, 0, 0, 0])

    def test_column_count(self):
        df = make_synthetic_patient()
        ts = create_time_since_features(df, INPUT_FEATURES)
        assert ts.shape[1] == 40


class TestImputeRemaining:
    def test_computes_stats_and_fills(self):
        df = make_synthetic_patient()
        filled = forward_fill_patient(df)
        result, stats = impute_remaining(filled, INPUT_FEATURES)
        assert not result[INPUT_FEATURES].isna().any().any()
        assert isinstance(stats, dict)

    def test_uses_provided_stats(self):
        df = make_synthetic_patient()
        filled = forward_fill_patient(df)
        result, _ = impute_remaining(filled, ["Temp"], train_stats={"Temp": 36.5})
        assert result["Temp"].iloc[0] == 36.5
        assert result["Temp"].iloc[1] == 36.5
        assert result["Temp"].iloc[2] == 37.0


class TestNormalizeFeatures:
    def test_zero_mean(self):
        df = make_synthetic_patient()
        result, stats = normalize_features(df, ["Resp"])
        assert abs(result["Resp"].mean()) < 1e-10
        assert "Resp" in stats
        assert "mean" in stats["Resp"]

    def test_uses_provided_stats(self):
        df = make_synthetic_patient()
        stats = {"Resp": {"mean": 20.0, "std": 2.0, "p1": 16.0, "p99": 24.0}}
        result, _ = normalize_features(df, ["Resp"], stats=stats)
        assert result["Resp"].iloc[0] == pytest.approx(-1.0)
        assert result["Resp"].iloc[2] == pytest.approx(1.0)

    def test_constant_column_no_error(self):
        df = make_synthetic_patient()
        result, stats = normalize_features(df, ["Age"])
        assert not result["Age"].isna().any()
        assert stats["Age"]["std"] == 1.0


class TestPreprocessPatient:
    def test_output_shape(self):
        df = make_synthetic_patient()
        result, stats = preprocess_patient(df)
        # 40 values + 40 masks + 40 time_since = 120
        assert result.shape == (5, N_TOTAL_FEATURES)
        assert result.shape[1] == 120
        assert stats is not None
        assert "impute" in stats
        assert "normalize" in stats

    def test_no_nans_in_output(self):
        df = make_synthetic_patient()
        result, _ = preprocess_patient(df)
        assert not result.isna().any().any()

    def test_column_ordering(self):
        df = make_synthetic_patient()
        result, _ = preprocess_patient(df)
        cols = list(result.columns)
        # First 40: raw feature names
        assert cols[:40] == INPUT_FEATURES
        # Next 40: *_missing
        assert cols[40:80] == [f"{c}_missing" for c in INPUT_FEATURES]
        # Last 40: *_time_since
        assert cols[80:120] == [f"{c}_time_since" for c in INPUT_FEATURES]

    def test_with_provided_stats(self):
        df = make_synthetic_patient()
        _, stats = preprocess_patient(df)
        result, returned_stats = preprocess_patient(df, train_stats=stats)
        assert result.shape == (5, 120)
        assert returned_stats is None

    def test_with_real_patients(self):
        """Smoke test on 10 real patients from training_setA."""
        patients = load_all_patients(DATA_DIR, max_patients=10)
        for pid, df in patients.items():
            result, _ = preprocess_patient(df)
            assert result.shape[1] == 120, f"Wrong column count for {pid}"
            assert not result.isna().any().any(), f"NaNs remain for {pid}"
            assert len(result) == len(df), f"Row count changed for {pid}"

    def test_short_patient(self):
        """Verify pipeline handles a 1-hour patient without errors."""
        data = {col: [np.nan] for col in ALL_FEATURES}
        data["HR"] = [80.0]
        data["Resp"] = [18.0]
        data["SBP"] = [120.0]
        data["DBP"] = [70.0]
        data["MAP"] = [80.0]
        data["Age"] = [50.0]
        data["ICULOS"] = [1]
        data["SepsisLabel"] = [0]
        df = pd.DataFrame(data)
        result, _ = preprocess_patient(df)
        assert result.shape == (1, 120)
        assert not result.isna().any().any()
