"""LightGBM baseline model for sepsis prediction."""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.preprocessing import VITAL_SIGNS
from src.training.trainer import compute_utility_score


def _compute_rolling(base, vital_indices, n_hours, window):
    """Compute rolling mean/min/max/std for vital signs over a given window."""
    n_vitals = len(vital_indices)
    r_mean = np.zeros((n_hours, n_vitals), dtype=np.float32)
    r_min = np.zeros((n_hours, n_vitals), dtype=np.float32)
    r_max = np.zeros((n_hours, n_vitals), dtype=np.float32)
    r_std = np.zeros((n_hours, n_vitals), dtype=np.float32)

    for t in range(n_hours):
        start = max(0, t - window + 1)
        for j, vi in enumerate(vital_indices):
            segment = base[start:t + 1, vi]
            r_mean[t, j] = segment.mean()
            r_min[t, j] = segment.min()
            r_max[t, j] = segment.max()
            r_std[t, j] = segment.std() if len(segment) > 1 else 0.0

    return r_mean, r_min, r_max, r_std


def create_baseline_features(patient_df, feature_columns, enhanced=False):
    """Create expanded feature set for a single patient.

    For each hour, produces:
      - Current values (the 120 preprocessed features)
      - Rolling stats over last 6 hours: mean, min, max, std for vital signs
      - Trend features: linear regression slope over last 6 hours for vital signs

    If enhanced=True, also adds:
      - 12-hour and 24-hour rolling windows (mean, min, max, std)
      - Rate of change (current - 6h ago) for vital signs
      - Time-weighted averages (exponential decay) for vital signs
      - Clinical interaction features (shock index, MAP trend, etc.)

    Args:
        patient_df: Preprocessed DataFrame with 120 feature columns + SepsisLabel.
        feature_columns: List of the 120 feature column names.
        enhanced: If True, compute additional engineered features.

    Returns:
        2D numpy array (n_hours, n_expanded_features).
    """
    n_hours = len(patient_df)
    base = patient_df[feature_columns].values.astype(np.float32)

    vital_cols = [c for c in feature_columns if c in VITAL_SIGNS]
    vital_indices = [feature_columns.index(c) for c in vital_cols]
    n_vitals = len(vital_indices)

    # 6-hour rolling stats
    r6_mean, r6_min, r6_max, r6_std = _compute_rolling(base, vital_indices, n_hours, 6)

    # 6-hour trend (slope)
    trend6 = np.zeros((n_hours, n_vitals), dtype=np.float32)
    for t in range(n_hours):
        start = max(0, t - 5)
        for j, vi in enumerate(vital_indices):
            segment = base[start:t + 1, vi]
            if len(segment) >= 2:
                x = np.arange(len(segment), dtype=np.float32)
                x_mean = x.mean()
                y_mean = segment.mean()
                denom = ((x - x_mean) ** 2).sum()
                if denom > 0:
                    trend6[t, j] = ((x - x_mean) * (segment - y_mean)).sum() / denom

    parts = [base, r6_mean, r6_min, r6_max, r6_std, trend6]

    if enhanced:
        # 12-hour and 24-hour rolling windows
        r12_mean, r12_min, r12_max, r12_std = _compute_rolling(
            base, vital_indices, n_hours, 12,
        )
        r24_mean, r24_min, r24_max, r24_std = _compute_rolling(
            base, vital_indices, n_hours, 24,
        )
        parts.extend([r12_mean, r12_min, r12_max, r12_std])
        parts.extend([r24_mean, r24_min, r24_max, r24_std])

        # Rate of change: current value - value 6 hours ago
        rate_of_change = np.zeros((n_hours, n_vitals), dtype=np.float32)
        for t in range(n_hours):
            t_prev = max(0, t - 6)
            for j, vi in enumerate(vital_indices):
                rate_of_change[t, j] = base[t, vi] - base[t_prev, vi]
        parts.append(rate_of_change)

        # Time-weighted average (exponential decay, half-life = 3 hours)
        tw_avg = np.zeros((n_hours, n_vitals), dtype=np.float32)
        decay = np.exp(-np.log(2) / 3.0)  # half-life of 3 hours
        for j, vi in enumerate(vital_indices):
            running = base[0, vi]
            for t in range(n_hours):
                running = decay * running + (1 - decay) * base[t, vi]
                tw_avg[t, j] = running
        parts.append(tw_avg)

        # Clinical interaction features
        hr_idx = feature_columns.index("HR") if "HR" in feature_columns else None
        sbp_idx = feature_columns.index("SBP") if "SBP" in feature_columns else None
        map_idx = feature_columns.index("MAP") if "MAP" in feature_columns else None
        resp_idx = feature_columns.index("Resp") if "Resp" in feature_columns else None
        temp_idx = feature_columns.index("Temp") if "Temp" in feature_columns else None

        interactions = []

        # Shock Index: HR * SBP product (avoids division on z-scored data)
        if hr_idx is not None and sbp_idx is not None:
            shock_feat = base[:, hr_idx:hr_idx + 1] * base[:, sbp_idx:sbp_idx + 1]
            interactions.append(shock_feat)

        # MAP trend: MAP_current - MAP_6h_ago
        if map_idx is not None:
            map_trend = np.zeros((n_hours, 1), dtype=np.float32)
            for t in range(n_hours):
                t_prev = max(0, t - 6)
                map_trend[t, 0] = base[t, map_idx] - base[t_prev, map_idx]
            interactions.append(map_trend)

        # Resp * HR product (avoids division on z-scored data)
        if resp_idx is not None and hr_idx is not None:
            resp_hr = base[:, resp_idx:resp_idx + 1] * base[:, hr_idx:hr_idx + 1]
            interactions.append(resp_hr)

        # Temperature deviation (absolute value of z-scored temp)
        if temp_idx is not None:
            temp_dev = np.abs(base[:, temp_idx:temp_idx + 1])
            interactions.append(temp_dev)

        if interactions:
            parts.extend(interactions)

    return np.hstack(parts)


def get_baseline_feature_names(feature_columns, enhanced=False):
    """Return names for the expanded baseline feature set."""
    vital_cols = [c for c in feature_columns if c in VITAL_SIGNS]
    names = list(feature_columns)

    # 6-hour rolling stats + trend (always included)
    for stat in ["rolling_mean", "rolling_min", "rolling_max", "rolling_std", "trend"]:
        names.extend([f"{c}_{stat}" for c in vital_cols])

    if enhanced:
        for window in [12, 24]:
            for stat in ["rolling_mean", "rolling_min", "rolling_max", "rolling_std"]:
                names.extend([f"{c}_{stat}_{window}h" for c in vital_cols])

        names.extend([f"{c}_rate_of_change" for c in vital_cols])
        names.extend([f"{c}_tw_avg" for c in vital_cols])

        # Clinical interactions (only if vitals exist in feature_columns)
        if "HR" in vital_cols and "SBP" in vital_cols:
            names.append("HR_SBP_product")
        if "MAP" in vital_cols:
            names.append("MAP_trend_6h")
        if "Resp" in vital_cols and "HR" in vital_cols:
            names.append("Resp_HR_product")
        if "Temp" in vital_cols:
            names.append("temp_deviation")

    return names


def prepare_baseline_data(patients_dict, feature_columns, enhanced=False):
    """Flatten patient dict into arrays for LightGBM.

    Args:
        patients_dict: Dict of preprocessed DataFrames (120 features + SepsisLabel).
        feature_columns: List of the 120 feature column names.
        enhanced: If True, compute additional engineered features.

    Returns:
        (X, y, patient_ids, hour_indices) where X is (n_samples, n_features).
    """
    all_X, all_y, all_pids, all_hours = [], [], [], []

    for pid, df in patients_dict.items():
        X_patient = create_baseline_features(df, feature_columns, enhanced=enhanced)
        y_patient = df["SepsisLabel"].values.astype(np.float32)

        all_X.append(X_patient)
        all_y.append(y_patient)
        all_pids.extend([pid] * len(df))
        all_hours.extend(range(len(df)))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Validate and clean features
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {n_nan:,} NaN and {n_inf:,} Inf values found, replacing...")
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clip extreme values to prevent LightGBM issues
    np.clip(X, -1e6, 1e6, out=X)

    print(f"  Feature stats: min={X.min():.2f}, max={X.max():.2f}, "
          f"mean={X.mean():.4f}, std={X.std():.4f}")

    return (X, y, all_pids, all_hours)


class LightGBMBaseline:
    """LightGBM classifier tuned for imbalanced sepsis prediction."""

    DEFAULT_PARAMS = {
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

    IMPROVED_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "is_unbalance": True,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    def __init__(self, params=None, improved=False):
        self.params = dict(self.IMPROVED_PARAMS if improved else self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.model = None
        self.feature_names = None
        self.enhanced = improved

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=None,
            early_stopping_rounds=None, feature_names=None):
        """Train the model with optional early stopping on validation set.

        For improved/enhanced mode, early stopping is disabled and a fixed
        number of trees (1000) are trained. Validation loss is monitored
        for logging only.
        """
        if num_boost_round is None:
            num_boost_round = 1000 if self.enhanced else 1000
        if early_stopping_rounds is None:
            early_stopping_rounds = None if self.enhanced else 50

        self.feature_names = feature_names
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
            # Only add early stopping for non-enhanced mode
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))

        es_str = str(early_stopping_rounds) if early_stopping_rounds else "disabled"
        print(f"  LightGBM params: leaves={self.params['num_leaves']}, "
              f"lr={self.params['learning_rate']}, rounds={num_boost_round}, "
              f"early_stop={es_str}")

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        print(f"  Trained {self.model.num_trees()} trees")

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
                "enhanced": self.enhanced,
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


def train_and_evaluate_baseline(train_patients, val_patients, feature_columns,
                                improved=False):
    """End-to-end baseline training and evaluation.

    Args:
        train_patients: Dict of preprocessed training patient DataFrames.
        val_patients: Dict of preprocessed validation patient DataFrames.
        feature_columns: List of 120 feature column names.
        improved: If True, use enhanced features and improved hyperparameters.

    Returns:
        (model, best_threshold, utility_score, feature_importance_df)
    """
    feature_names = get_baseline_feature_names(feature_columns, enhanced=improved)

    print("Preparing training data...")
    X_train, y_train, _, _ = prepare_baseline_data(
        train_patients, feature_columns, enhanced=improved,
    )
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Positive rate: {y_train.mean():.4f}")

    print("Preparing validation data...")
    X_val, y_val, val_pids, val_hours = prepare_baseline_data(
        val_patients, feature_columns, enhanced=improved,
    )
    print(f"  Val: {X_val.shape[0]:,} samples")

    mode = "improved" if improved else "default"
    print(f"Training LightGBM ({mode} params)...")
    model = LightGBMBaseline(improved=improved)
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
