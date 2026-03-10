"""Ensemble and post-processing utilities for sepsis prediction."""

import numpy as np


# ICULOS is the last of the 6 demographics features (index 39 of 40 raw features).
# In the 120-feature vector: raw values occupy indices 0-39.
ICULOS_FEATURE_INDEX = 39


def filter_by_iculos(probs, hour_indices, iculos_threshold=6):
    """Zero out predictions where ICULOS < threshold.

    Removes false alarms during early ICU admission where the model
    tends to trigger on admission patterns rather than sepsis signals.

    Args:
        probs: Array of prediction probabilities, shape (n,).
        hour_indices: 0-based hour index per sample (ICULOS = hour + 1).
        iculos_threshold: Minimum ICULOS (1-indexed) to allow predictions.

    Returns:
        Filtered probabilities (copy).
    """
    filtered = np.array(probs, dtype=np.float64)
    # hour_indices are 0-based; ICULOS is 1-based, so ICULOS = hour + 1
    for i, hour in enumerate(hour_indices):
        if (hour + 1) < iculos_threshold:
            filtered[i] = 0.0
    return filtered


class SepsisEnsemble:
    """Weighted ensemble of GRU and LightGBM predictions."""

    def __init__(self, gru_weight=0.6):
        self.gru_weight = gru_weight

    def combine(self, p_gru, p_lgb):
        """Weighted average of GRU and LightGBM probabilities.

        Args:
            p_gru: GRU probabilities, shape (n,).
            p_lgb: LightGBM probabilities, shape (n,).

        Returns:
            Combined probabilities, shape (n,).
        """
        p_gru = np.asarray(p_gru, dtype=np.float64)
        p_lgb = np.asarray(p_lgb, dtype=np.float64)
        return self.gru_weight * p_gru + (1 - self.gru_weight) * p_lgb

    @staticmethod
    def combine_max(p_gru, p_lgb):
        """Element-wise max of GRU and LightGBM probabilities."""
        return np.maximum(p_gru, p_lgb)

    @staticmethod
    def combine_or(p_gru, p_lgb, gru_thresh=0.3, lgb_thresh=0.3):
        """Union: positive if either model exceeds its threshold.

        Returns probability = max of the two (to allow threshold search).
        """
        p_gru = np.asarray(p_gru, dtype=np.float64)
        p_lgb = np.asarray(p_lgb, dtype=np.float64)
        either_positive = (p_gru >= gru_thresh) | (p_lgb >= lgb_thresh)
        # Return max prob where either fires, 0 otherwise
        result = np.where(either_positive, np.maximum(p_gru, p_lgb), 0.0)
        return result
