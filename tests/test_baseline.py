"""Tests for LightGBM baseline model."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import VITAL_SIGNS
from src.models.baseline import (
    LightGBMBaseline,
    create_baseline_features,
    get_baseline_feature_names,
    prepare_baseline_data,
    train_and_evaluate_baseline,
)


N_RAW_FEATURES = 6
FEATURE_COLS = ["HR", "O2Sat", "Temp", "f3", "f4", "f5"]


def _make_patient_df(n_hours=20):
    """Create a synthetic preprocessed patient DataFrame."""
    data = np.random.randn(n_hours, N_RAW_FEATURES).astype(np.float32)
    df = pd.DataFrame(data, columns=FEATURE_COLS)
    df["SepsisLabel"] = 0.0
    df.loc[n_hours - 2:, "SepsisLabel"] = 1.0
    return df


def _make_patients(n_patients=10, n_hours=15, sepsis_ratio=0.2):
    """Create a dict of synthetic patients."""
    patients = {}
    for i in range(n_patients):
        df = _make_patient_df(n_hours)
        if i >= int(n_patients * sepsis_ratio):
            df["SepsisLabel"] = 0.0
        patients[f"p{i}"] = df
    return patients


class TestCreateBaselineFeatures:
    def test_output_shape(self):
        df = _make_patient_df(20)
        features = create_baseline_features(df, FEATURE_COLS)
        # 3 vitals in FEATURE_COLS (HR, O2Sat, Temp) * 5 stats + 6 base = 21
        n_vitals = len([c for c in FEATURE_COLS if c in VITAL_SIGNS])
        expected_cols = N_RAW_FEATURES + n_vitals * 5
        assert features.shape == (20, expected_cols)

    def test_no_nans(self):
        df = _make_patient_df(10)
        features = create_baseline_features(df, FEATURE_COLS)
        assert not np.isnan(features).any()

    def test_single_hour(self):
        df = _make_patient_df(1)
        features = create_baseline_features(df, FEATURE_COLS)
        assert features.shape[0] == 1
        assert not np.isnan(features).any()

    def test_feature_names_match_shape(self):
        df = _make_patient_df(10)
        features = create_baseline_features(df, FEATURE_COLS)
        names = get_baseline_feature_names(FEATURE_COLS)
        assert len(names) == features.shape[1]


class TestPrepareBaselineData:
    def test_output_shapes(self):
        patients = _make_patients(5, n_hours=10)
        X, y, pids, hours = prepare_baseline_data(patients, FEATURE_COLS)
        assert X.shape[0] == 50  # 5 patients * 10 hours
        assert y.shape == (50,)
        assert len(pids) == 50
        assert len(hours) == 50

    def test_labels_match(self):
        patients = _make_patients(3, n_hours=8)
        _, y, _, _ = prepare_baseline_data(patients, FEATURE_COLS)
        # Check total positive labels match what we put in
        total_pos = sum(df["SepsisLabel"].sum() for df in patients.values())
        assert y.sum() == pytest.approx(total_pos)


class TestLightGBMBaseline:
    def test_train_predict(self):
        n_features = 21
        X_train = np.random.randn(200, n_features).astype(np.float32)
        y_train = np.zeros(200, dtype=np.float32)
        y_train[:20] = 1.0

        model = LightGBMBaseline({"num_leaves": 8, "n_estimators": 10})
        model.fit(X_train, y_train, num_boost_round=10)

        probs = model.predict_proba(X_train)
        assert probs.shape == (200,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_train_with_early_stopping(self):
        n_features = 15
        X_train = np.random.randn(200, n_features).astype(np.float32)
        y_train = np.zeros(200, dtype=np.float32)
        y_train[:20] = 1.0
        X_val = np.random.randn(50, n_features).astype(np.float32)
        y_val = np.zeros(50, dtype=np.float32)
        y_val[:5] = 1.0

        model = LightGBMBaseline({"num_leaves": 8})
        model.fit(X_train, y_train, X_val, y_val, num_boost_round=100, early_stopping_rounds=5)

        probs = model.predict_proba(X_val)
        assert probs.shape == (50,)

    def test_feature_importance(self):
        n_features = 10
        names = [f"feat_{i}" for i in range(n_features)]
        X = np.random.randn(100, n_features).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = LightGBMBaseline({"num_leaves": 8})
        model.fit(X, y, num_boost_round=20, feature_names=names)

        imp = model.feature_importance()
        assert len(imp) == n_features
        assert list(imp.columns) == ["feature", "importance"]
        assert imp["feature"].iloc[0] == "feat_0"  # most important

    def test_save_load(self, tmp_path):
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = LightGBMBaseline({"num_leaves": 4})
        model.fit(X, y, num_boost_round=10, feature_names=["a", "b", "c", "d", "e"])

        path = str(tmp_path / "model.pkl")
        model.save(path)

        loaded = LightGBMBaseline.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X), loaded.predict_proba(X),
        )


class TestTrainAndEvaluate:
    def test_end_to_end(self):
        patients = _make_patients(20, n_hours=12, sepsis_ratio=0.3)
        train = {k: v for i, (k, v) in enumerate(patients.items()) if i < 14}
        val = {k: v for i, (k, v) in enumerate(patients.items()) if i >= 14}

        model, threshold, utility, importance = train_and_evaluate_baseline(
            train, val, FEATURE_COLS,
        )

        assert model.model is not None
        assert 0.1 <= threshold <= 0.9
        assert isinstance(utility, float)
        assert len(importance) > 0
