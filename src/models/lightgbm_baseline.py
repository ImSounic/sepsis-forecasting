"""Original LightGBM baseline model for sepsis prediction.

Simple model with 160 features (120 base + 40 rolling/trend).
Preserved for comparison against the improved LightGBM.
"""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.preprocessing import VITAL_SIGNS
from src.training.trainer import compute_utility_score


def create_baseline_features(patient_df, feature_columns):
    """Create 160-feature set for a single patient.

    For each hour, produces:
      - Current values (the 120 preprocessed features)
      - Rolling stats over last 6 hours: mean, min, max, std for 8 vital signs
      - Trend features: linear regression slope over last 6 hours for 8 vital signs

    Total: 120 base + 8*5 rolling = 160 features.
    """
    n_hours = len(patient_df)
    base = patient_df[feature_columns].values.astype(np.float32)

    vital_cols = [c for c in feature_columns if c in VITAL_SIGNS]
    vital_indices = [feature_columns.index(c) for c in vital_cols]
    n_vitals = len(vital_indices)
    window = 6

    rolling_mean = np.zeros((n_hours, n_vitals), dtype=np.float32)
    rolling_min = np.zeros((n_hours, n_vitals), dtype=np.float32)
    rolling_max = np.zeros((n_hours, n_vitals), dtype=np.float32)
    rolling_std = np.zeros((n_hours, n_vitals), dtype=np.float32)
    trend = np.zeros((n_hours, n_vitals), dtype=np.float32)

    for t in range(n_hours):
        start = max(0, t - window + 1)
        for j, vi in enumerate(vital_indices):
            segment = base[start:t + 1, vi]
            rolling_mean[t, j] = segment.mean()
            rolling_min[t, j] = segment.min()
            rolling_max[t, j] = segment.max()
            rolling_std[t, j] = segment.std() if len(segment) > 1 else 0.0

            if len(segment) >= 2:
                x = np.arange(len(segment), dtype=np.float32)
                x_mean = x.mean()
                y_mean = segment.mean()
                denom = ((x - x_mean) ** 2).sum()
                if denom > 0:
                    trend[t, j] = ((x - x_mean) * (segment - y_mean)).sum() / denom

    return np.hstack([base, rolling_mean, rolling_min, rolling_max, rolling_std, trend])


def get_baseline_feature_names(feature_columns):
    """Return 160 feature names for the baseline feature set."""
    vital_cols = [c for c in feature_columns if c in VITAL_SIGNS]
    names = list(feature_columns)
    for stat in ["rolling_mean", "rolling_min", "rolling_max", "rolling_std", "trend"]:
        names.extend([f"{c}_{stat}" for c in vital_cols])
    return names


def prepare_baseline_data(patients_dict, feature_columns):
    """Flatten patient dict into arrays for LightGBM.

    Returns:
        (X, y, patient_ids, hour_indices) where X is (n_samples, 160).
    """
    all_X, all_y, all_pids, all_hours = [], [], [], []

    for pid, df in patients_dict.items():
        X_patient = create_baseline_features(df, feature_columns)
        y_patient = df["SepsisLabel"].values.astype(np.float32)

        all_X.append(X_patient)
        all_y.append(y_patient)
        all_pids.extend([pid] * len(df))
        all_hours.extend(range(len(df)))

    return (
        np.vstack(all_X),
        np.concatenate(all_y),
        all_pids,
        all_hours,
    )


class LightGBMBaseline:
    """Original LightGBM classifier with default hyperparameters."""

    def __init__(self, params=None):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "is_unbalance": True,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        if params:
            self.params.update(params)
        self.model = None
        self.feature_names = None
        self.enhanced = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=1000,
            early_stopping_rounds=50, feature_names=None):
        """Train with early stopping on validation set."""
        self.feature_names = feature_names
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict_proba(self, X):
        """Return predicted probabilities."""
        return self.model.predict(X)

    def feature_importance(self):
        """Return feature importance as a DataFrame sorted by importance."""
        names = self.feature_names or [f"f{i}" for i in range(self.model.num_feature())]
        importance = self.model.feature_importance(importance_type="gain")
        df = pd.DataFrame({"feature": names, "importance": importance})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path):
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "enhanced": False,
            }, f)

    @classmethod
    def load(cls, path):
        """Load model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.enhanced = data.get("enhanced", False)
        return obj


def train_and_evaluate_baseline(train_patients, val_patients, feature_columns):
    """End-to-end baseline training and evaluation with 160 features.

    Returns:
        (model, best_threshold, utility_score, feature_importance_df)
    """
    feature_names = get_baseline_feature_names(feature_columns)

    print("Preparing training data...")
    X_train, y_train, _, _ = prepare_baseline_data(train_patients, feature_columns)
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Positive rate: {y_train.mean():.4f}")

    print("Preparing validation data...")
    X_val, y_val, val_pids, val_hours = prepare_baseline_data(val_patients, feature_columns)
    print(f"  Val: {X_val.shape[0]:,} samples")

    print("Training LightGBM (baseline params)...")
    model = LightGBMBaseline()
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    print("Finding optimal threshold...")
    val_probs = model.predict_proba(X_val)

    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.arange(0.1, 0.91, 0.05):
        binary_preds = (val_probs >= threshold).astype(int).tolist()
        score = compute_utility_score(
            y_val.tolist(), binary_preds, val_pids, val_hours,
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    print(f"Best threshold: {best_threshold:.2f}, Utility: {best_score:.4f}")

    importance_df = model.feature_importance()

    return model, best_threshold, best_score, importance_df
