"""Feature engineering pipeline for PhysioNet 2019 Challenge data.

Transforms raw patient DataFrames into model-ready features:
  40 values + 40 missingness masks + 40 time-since = 120 input features.
"""

import numpy as np
import pandas as pd

from src.data.preprocessing import CLINICAL_FEATURES, DEMOGRAPHICS

# All 40 input features (everything except SepsisLabel)
INPUT_FEATURES = CLINICAL_FEATURES + DEMOGRAPHICS

N_TOTAL_FEATURES = len(INPUT_FEATURES) + 2 * len(INPUT_FEATURES)  # 120


def forward_fill_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values within a single patient's time series.

    Only fills forward in time (no look-ahead). Leading NaNs before the first
    observation remain NaN and are handled by impute_remaining.
    """
    return df.ffill()


def create_missingness_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Create binary missingness masks: 1 = originally missing, 0 = observed."""
    mask = df[feature_columns].isna().astype(int)
    mask.columns = [f"{col}_missing" for col in feature_columns]
    return mask


def create_time_since_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Create hours-since-last-observation features.

    At each observed value the counter resets to 0.
    At each missing value the counter increments by 1.
    Before the first observation, counts up from 1.
    """
    n = len(df)
    time_since = pd.DataFrame(index=df.index)

    for col in feature_columns:
        is_observed = df[col].notna().values
        ts = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if is_observed[i]:
                ts[i] = 0.0
            elif i == 0:
                ts[i] = 1.0
            else:
                ts[i] = ts[i - 1] + 1.0
        time_since[f"{col}_time_since"] = ts

    return time_since


def impute_remaining(
    df: pd.DataFrame,
    feature_columns: list[str],
    strategy: str = "median",
    train_stats: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """Impute remaining NaNs after forward-fill with population statistics.

    Args:
        df: Patient DataFrame (after forward-fill).
        feature_columns: Columns to impute.
        strategy: 'median' or 'mean' (only used when computing stats).
        train_stats: Pre-computed {feature: fill_value} dict. If None, computes
            from the data and returns the computed stats.

    Returns:
        (imputed_df, stats_dict)
    """
    result = df.copy()
    compute = train_stats is None
    stats = {} if compute else train_stats

    for col in feature_columns:
        if col not in result.columns:
            continue
        if compute:
            if strategy == "median":
                fill_val = result[col].median()
            else:
                fill_val = result[col].mean()
            if pd.isna(fill_val):
                fill_val = 0.0
            stats[col] = fill_val
        else:
            fill_val = stats.get(col, 0.0)
        result[col] = result[col].fillna(fill_val)

    return result, stats


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    stats: dict = None,
) -> tuple[pd.DataFrame, dict]:
    """Z-score normalization with winsorizing at 1st/99th percentile.

    Args:
        df: Patient DataFrame (after imputation, no NaNs expected).
        feature_columns: Columns to normalize.
        stats: Pre-computed {feature: {mean, std, p1, p99}} dict. If None,
            computes from the data and returns the computed stats.

    Returns:
        (normalized_df, stats_dict)
    """
    result = df.copy()
    compute = stats is None
    norm_stats = {} if compute else stats

    for col in feature_columns:
        if col not in result.columns:
            continue
        if compute:
            p1 = float(result[col].quantile(0.01))
            p99 = float(result[col].quantile(0.99))
            if pd.isna(p1) or pd.isna(p99):
                p1, p99 = 0.0, 0.0
            clipped = result[col].clip(p1, p99)
            mean_val = float(clipped.mean())
            std_val = float(clipped.std())
            if pd.isna(mean_val):
                mean_val = 0.0
            if std_val == 0 or pd.isna(std_val):
                std_val = 1.0
            norm_stats[col] = {"mean": mean_val, "std": std_val, "p1": p1, "p99": p99}
        else:
            if col not in norm_stats:
                continue
            s = norm_stats[col]
            mean_val, std_val = s["mean"], s["std"]
            p1, p99 = s["p1"], s["p99"]

        result[col] = (result[col].clip(p1, p99) - mean_val) / std_val

    return result, norm_stats


def preprocess_patient(
    df: pd.DataFrame, train_stats: dict = None
) -> tuple[pd.DataFrame, dict | None]:
    """Full preprocessing pipeline for a single patient.

    Steps:
      1. Capture missingness/time-since (before filling)
      2. Forward-fill -> impute remaining NaNs
      3. Normalize value features (original 40)
      4. Concatenate: values + masks + time_since

    Args:
        df: Raw patient DataFrame (41 columns including SepsisLabel).
        train_stats: Dict with 'impute' and 'normalize' keys from training set.
            If None, computes stats from this patient (for testing or stat collection).

    Returns:
        (processed_df, stats_or_None) where processed_df has 120 columns:
        40 values + 40 missingness masks + 40 time-since.
    """
    feature_cols = INPUT_FEATURES

    # Capture missingness and time-since BEFORE any filling
    mask_df = create_missingness_features(df, feature_cols)
    time_since_df = create_time_since_features(df, feature_cols)

    # Forward-fill then impute remaining NaNs
    filled = forward_fill_patient(df)
    impute_stats = train_stats.get("impute") if train_stats else None
    filled, computed_impute = impute_remaining(filled, feature_cols, train_stats=impute_stats)

    # Normalize value features
    norm_stats = train_stats.get("normalize") if train_stats else None
    normalized, computed_norm = normalize_features(
        filled, feature_cols, stats=norm_stats
    )

    # Final output: 40 normalized values + 40 masks + 40 time_since = 120
    result = pd.concat(
        [normalized[feature_cols], mask_df, time_since_df], axis=1
    )

    if train_stats is None:
        return result, {"impute": computed_impute, "normalize": computed_norm}
    return result, None
