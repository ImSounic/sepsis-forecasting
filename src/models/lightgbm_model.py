"""Improved LightGBM model for sepsis prediction.

Enhanced version with 244 features (120 base + 124 engineered) and
tuned hyperparameters. For the original 160-feature baseline, see
lightgbm_baseline.py.
"""

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.preprocessing import VITAL_SIGNS, LAB_VALUES
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


def _unnormalize(z_values, col_name, norm_stats):
    """Convert z-scored values back to raw scale using normalization stats."""
    if norm_stats is None or col_name not in norm_stats:
        return z_values
    s = norm_stats[col_name]
    return z_values * s["std"] + s["mean"]


def _sofa_score_components(raw_values, feature_columns, norm_stats):
    """Compute SOFA-based clinical scores from raw (un-normalized) values.

    Returns array of shape (n_hours, 7): MAP, Creatinine, Bilirubin, Platelets,
    Resp/FiO2, coagulation, and total partial SOFA.
    """
    n_hours = raw_values.shape[0]
    scores = np.zeros((n_hours, 7), dtype=np.float32)

    def _get_raw(col):
        if col not in feature_columns:
            return None
        idx = feature_columns.index(col)
        return _unnormalize(raw_values[:, idx], col, norm_stats)

    # MAP component (0-4): <70=1, dopamine<=5=2, dopa>5=3, dopa>15=4
    map_raw = _get_raw("MAP")
    if map_raw is not None:
        scores[:, 0] = np.where(map_raw < 70, 1, 0)

    # Creatinine component (0-4): 1.2-1.9=1, 2.0-3.4=2, 3.5-4.9=3, >5=4
    creat_raw = _get_raw("Creatinine")
    if creat_raw is not None:
        scores[:, 1] = np.where(creat_raw >= 5.0, 4,
                       np.where(creat_raw >= 3.5, 3,
                       np.where(creat_raw >= 2.0, 2,
                       np.where(creat_raw >= 1.2, 1, 0))))

    # Bilirubin component (0-4): 1.2-1.9=1, 2.0-5.9=2, 6.0-11.9=3, >12=4
    bili_raw = _get_raw("Bilirubin_total")
    if bili_raw is not None:
        scores[:, 2] = np.where(bili_raw >= 12.0, 4,
                       np.where(bili_raw >= 6.0, 3,
                       np.where(bili_raw >= 2.0, 2,
                       np.where(bili_raw >= 1.2, 1, 0))))

    # Platelets component (0-4): <150=1, <100=2, <50=3, <20=4
    plt_raw = _get_raw("Platelets")
    if plt_raw is not None:
        scores[:, 3] = np.where(plt_raw < 20, 4,
                       np.where(plt_raw < 50, 3,
                       np.where(plt_raw < 100, 2,
                       np.where(plt_raw < 150, 1, 0))))

    # Respiration: PaO2/FiO2 ratio if available
    pao2_raw = _get_raw("PaCO2")  # Note: dataset has PaCO2, not PaO2
    fio2_raw = _get_raw("FiO2")
    if fio2_raw is not None:
        # Use SaO2/FiO2 as proxy
        sao2_raw = _get_raw("SaO2")
        if sao2_raw is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(fio2_raw > 0, sao2_raw / fio2_raw, 0)
            scores[:, 4] = np.where(ratio < 100, 4,
                           np.where(ratio < 200, 3,
                           np.where(ratio < 300, 2,
                           np.where(ratio < 400, 1, 0))))

    # Total partial SOFA (sum of available components)
    scores[:, 6] = scores[:, :6].sum(axis=1)

    return scores


def _qsofa_score(raw_values, feature_columns, norm_stats):
    """Compute qSOFA score (0-3): SBP<=100, Resp>=22, altered mentation (not available).

    Returns array of shape (n_hours, 3): SBP component, Resp component, total.
    """
    n_hours = raw_values.shape[0]
    scores = np.zeros((n_hours, 3), dtype=np.float32)

    if "SBP" in feature_columns:
        idx = feature_columns.index("SBP")
        sbp_raw = _unnormalize(raw_values[:, idx], "SBP", norm_stats)
        scores[:, 0] = (sbp_raw <= 100).astype(np.float32)

    if "Resp" in feature_columns:
        idx = feature_columns.index("Resp")
        resp_raw = _unnormalize(raw_values[:, idx], "Resp", norm_stats)
        scores[:, 1] = (resp_raw >= 22).astype(np.float32)

    scores[:, 2] = scores[:, 0] + scores[:, 1]
    return scores


def _news_vital_scores(raw_values, feature_columns, norm_stats):
    """Compute NEWS-style vital sign scoring.

    Returns array of shape (n_hours, 4): HR score, Temp score, Resp score, total.
    """
    n_hours = raw_values.shape[0]
    scores = np.zeros((n_hours, 4), dtype=np.float32)

    # HR: 3 if <=40 or >=131; 2 if 111-130; 1 if 41-50 or 91-110
    if "HR" in feature_columns:
        idx = feature_columns.index("HR")
        hr = _unnormalize(raw_values[:, idx], "HR", norm_stats)
        scores[:, 0] = np.where((hr <= 40) | (hr >= 131), 3,
                       np.where((hr >= 111) & (hr <= 130), 2,
                       np.where(((hr >= 41) & (hr <= 50)) | ((hr >= 91) & (hr <= 110)), 1, 0)))

    # Temp: 3 if <=35; 2 if >=39.1; 1 if 35.1-36.0 or 38.1-39.0
    if "Temp" in feature_columns:
        idx = feature_columns.index("Temp")
        temp = _unnormalize(raw_values[:, idx], "Temp", norm_stats)
        scores[:, 1] = np.where(temp <= 35.0, 3,
                       np.where(temp >= 39.1, 2,
                       np.where(((temp >= 35.1) & (temp <= 36.0)) | ((temp >= 38.1) & (temp <= 39.0)), 1, 0)))

    # Resp: 3 if <8 or >25; 2 if 21-24; 1 if 9-11
    if "Resp" in feature_columns:
        idx = feature_columns.index("Resp")
        resp = _unnormalize(raw_values[:, idx], "Resp", norm_stats)
        scores[:, 2] = np.where((resp < 8) | (resp > 25), 3,
                       np.where((resp >= 21) & (resp <= 24), 2,
                       np.where((resp >= 9) & (resp <= 11), 1, 0)))

    scores[:, 3] = scores[:, :3].sum(axis=1)
    return scores


def _sirs_score(raw_values, feature_columns, norm_stats):
    """Compute SIRS score (0-4): Temp, HR, Resp/PaCO2, WBC components.

    Returns array of shape (n_hours, 5): Temp, HR, Resp, WBC components + total.
    """
    n_hours = raw_values.shape[0]
    scores = np.zeros((n_hours, 5), dtype=np.float32)

    def _get_raw(col):
        if col not in feature_columns:
            return None
        idx = feature_columns.index(col)
        return _unnormalize(raw_values[:, idx], col, norm_stats)

    # Temp: >38.0 or <36.0
    temp_raw = _get_raw("Temp")
    if temp_raw is not None:
        scores[:, 0] = ((temp_raw > 38.0) | (temp_raw < 36.0)).astype(np.float32)

    # HR: >90
    hr_raw = _get_raw("HR")
    if hr_raw is not None:
        scores[:, 1] = (hr_raw > 90).astype(np.float32)

    # Resp: >20 or PaCO2 < 32
    resp_raw = _get_raw("Resp")
    paco2_raw = _get_raw("PaCO2")
    if resp_raw is not None:
        resp_crit = (resp_raw > 20).astype(np.float32)
        if paco2_raw is not None:
            resp_crit = np.maximum(resp_crit, (paco2_raw < 32).astype(np.float32))
        scores[:, 2] = resp_crit

    # WBC: >12 or <4 (thousands)
    wbc_raw = _get_raw("WBC")
    if wbc_raw is not None:
        scores[:, 3] = ((wbc_raw > 12.0) | (wbc_raw < 4.0)).astype(np.float32)

    scores[:, 4] = scores[:, :4].sum(axis=1)
    return scores


def _mews_score(raw_values, feature_columns, norm_stats):
    """Compute Modified Early Warning Score (MEWS).

    Returns array of shape (n_hours, 5): SBP, HR, Resp, Temp components + total.
    """
    n_hours = raw_values.shape[0]
    scores = np.zeros((n_hours, 5), dtype=np.float32)

    def _get_raw(col):
        if col not in feature_columns:
            return None
        idx = feature_columns.index(col)
        return _unnormalize(raw_values[:, idx], col, norm_stats)

    # SBP: 0 (101-199), 1 (81-100), 2 (71-80 or >=200), 3 (<=70)
    sbp_raw = _get_raw("SBP")
    if sbp_raw is not None:
        scores[:, 0] = np.where(sbp_raw <= 70, 3,
                       np.where((sbp_raw <= 80) | (sbp_raw >= 200), 2,
                       np.where(sbp_raw <= 100, 1, 0)))

    # HR: 0 (51-100), 1 (41-50 or 101-110), 2 (<=40 or 111-129), 3 (>=130)
    hr_raw = _get_raw("HR")
    if hr_raw is not None:
        scores[:, 1] = np.where(hr_raw >= 130, 3,
                       np.where((hr_raw <= 40) | ((hr_raw >= 111) & (hr_raw <= 129)), 2,
                       np.where(((hr_raw >= 41) & (hr_raw <= 50)) | ((hr_raw >= 101) & (hr_raw <= 110)), 1, 0)))

    # Resp: 0 (9-14), 1 (15-20), 2 (21-29 or <9), 3 (>=30)
    resp_raw = _get_raw("Resp")
    if resp_raw is not None:
        scores[:, 2] = np.where(resp_raw >= 30, 3,
                       np.where((resp_raw < 9) | ((resp_raw >= 21) & (resp_raw <= 29)), 2,
                       np.where((resp_raw >= 15) & (resp_raw <= 20), 1, 0)))

    # Temp: 0 (35.0-38.4), 2 (<35.0 or >=38.5)
    temp_raw = _get_raw("Temp")
    if temp_raw is not None:
        scores[:, 3] = np.where((temp_raw < 35.0) | (temp_raw >= 38.5), 2, 0)

    scores[:, 4] = scores[:, :4].sum(axis=1)
    return scores


def _clinical_ratios(raw_values, feature_columns, norm_stats):
    """Compute clinical ratio features from raw values.

    Returns array of shape (n_hours, N) with: Shock Index, BUN/Creatinine,
    SaO2/FiO2, Pulse Pressure, Cardiac Output proxy.
    """
    n_hours = raw_values.shape[0]
    ratios = []

    def _get_raw(col):
        if col not in feature_columns:
            return None
        idx = feature_columns.index(col)
        return _unnormalize(raw_values[:, idx], col, norm_stats)

    # Shock Index: HR / SBP (normal ~0.5-0.7, elevated >0.9 indicates shock)
    hr_raw = _get_raw("HR")
    sbp_raw = _get_raw("SBP")
    if hr_raw is not None and sbp_raw is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            si = np.where(sbp_raw > 0, hr_raw / sbp_raw, 0)
        ratios.append(si.reshape(-1, 1))

    # BUN / Creatinine ratio (normal 10-20, elevated suggests pre-renal issues)
    bun_raw = _get_raw("BUN")
    creat_raw = _get_raw("Creatinine")
    if bun_raw is not None and creat_raw is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            bc = np.where(creat_raw > 0, bun_raw / creat_raw, 0)
        ratios.append(bc.reshape(-1, 1))

    # SaO2 / FiO2 ratio (oxygenation index)
    sao2_raw = _get_raw("SaO2")
    fio2_raw = _get_raw("FiO2")
    if sao2_raw is not None and fio2_raw is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = np.where(fio2_raw > 0, sao2_raw / fio2_raw, 0)
        ratios.append(sf.reshape(-1, 1))

    # Pulse Pressure: SBP - DBP (narrow PP indicates poor cardiac output)
    dbp_raw = _get_raw("DBP")
    if sbp_raw is not None and dbp_raw is not None:
        pp = sbp_raw - dbp_raw
        ratios.append(pp.reshape(-1, 1))

    # Cardiac Output proxy: (SBP - DBP) * HR
    if sbp_raw is not None and dbp_raw is not None and hr_raw is not None:
        co = (sbp_raw - dbp_raw) * hr_raw
        ratios.append(co.reshape(-1, 1))

    if ratios:
        return np.hstack(ratios)
    return np.zeros((n_hours, 0), dtype=np.float32)


def _differential_features(base, feature_columns):
    """Compute value changes between consecutive observations for all clinical features.

    Unlike rate_of_change (current - 6h ago), this computes the change between
    consecutive non-missing values, capturing short-term clinical deterioration.
    Uses the z-scored values directly (differences are scale-invariant).

    Returns array of shape (n_hours, n_clinical_features).
    """
    clinical_cols = [c for c in feature_columns if c in VITAL_SIGNS + LAB_VALUES]
    clinical_indices = [feature_columns.index(c) for c in clinical_cols]
    n_hours = base.shape[0]
    n_feats = len(clinical_indices)

    diffs = np.zeros((n_hours, n_feats), dtype=np.float32)
    for j, ci in enumerate(clinical_indices):
        # Use missingness mask if available (col_missing = 1 means was missing)
        mask_name = f"{clinical_cols[j]}_missing"
        if mask_name in feature_columns:
            mask_idx = feature_columns.index(mask_name)
            observed = base[:, mask_idx] < 0.5  # 0 = observed
        else:
            observed = np.ones(n_hours, dtype=bool)

        last_val = base[0, ci]
        for t in range(n_hours):
            if observed[t]:
                diffs[t, j] = base[t, ci] - last_val
                last_val = base[t, ci]
            # else: keep 0 (no change when not observed)

    return diffs


def create_baseline_features(patient_df, feature_columns, enhanced=False, norm_stats=None):
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
      - Clinical scores: SOFA components, qSOFA, NEWS (requires norm_stats)
      - Clinical ratios: Shock Index, BUN/Creatinine, SaO2/FiO2, Pulse Pressure
      - Differential features: consecutive observation changes for all clinical vars

    Args:
        patient_df: Preprocessed DataFrame with 120 feature columns + SepsisLabel.
        feature_columns: List of the 120 feature column names.
        enhanced: If True, compute additional engineered features.
        norm_stats: Dict of {feature: {mean, std, p1, p99}} for un-normalizing.
            Required for clinical score features (SOFA, qSOFA, NEWS, ratios).

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

        # --- New clinical features (require norm_stats for raw thresholds) ---

        # Differential features: consecutive observation changes
        diffs = _differential_features(base, feature_columns)
        parts.append(diffs)

        if norm_stats is not None:
            # SOFA score components (7 features)
            sofa = _sofa_score_components(base, feature_columns, norm_stats)
            parts.append(sofa)

            # qSOFA score (3 features)
            qsofa = _qsofa_score(base, feature_columns, norm_stats)
            parts.append(qsofa)

            # NEWS-style vital scores (4 features)
            news = _news_vital_scores(base, feature_columns, norm_stats)
            parts.append(news)

            # Clinical ratios from raw values (up to 5 features)
            ratios = _clinical_ratios(base, feature_columns, norm_stats)
            if ratios.shape[1] > 0:
                parts.append(ratios)

            # SIRS score (5 features)
            sirs = _sirs_score(base, feature_columns, norm_stats)
            parts.append(sirs)

            # MEWS score (5 features)
            mews = _mews_score(base, feature_columns, norm_stats)
            parts.append(mews)

        # --- Additional engineered features (no norm_stats required) ---

        # Multi-window rate of change: 3h and 12h
        for window, label in [(3, "3h"), (12, "12h")]:
            roc = np.zeros((n_hours, n_vitals), dtype=np.float32)
            for t in range(n_hours):
                t_prev = max(0, t - window)
                for j, vi in enumerate(vital_indices):
                    roc[t, j] = base[t, vi] - base[t_prev, vi]
            parts.append(roc)

        # Second EMA half-life (6 hours) for vital signs
        tw_avg_6h = np.zeros((n_hours, n_vitals), dtype=np.float32)
        decay_6h = np.exp(-np.log(2) / 6.0)
        for j, vi in enumerate(vital_indices):
            running = base[0, vi]
            for t in range(n_hours):
                running = decay_6h * running + (1 - decay_6h) * base[t, vi]
                tw_avg_6h[t, j] = running
        parts.append(tw_avg_6h)

        # Lab cross-interactions (on z-scored values)
        lab_interactions = []

        def _safe_idx(col):
            return feature_columns.index(col) if col in feature_columns else None

        lactate_idx = _safe_idx("Lactate")
        wbc_idx = _safe_idx("WBC")
        creat_idx = _safe_idx("Creatinine")
        bun_idx = _safe_idx("BUN")
        platelets_idx = _safe_idx("Platelets")

        # Lactate * HR
        if lactate_idx is not None and hr_idx is not None:
            lab_interactions.append(
                (base[:, lactate_idx] * base[:, hr_idx]).reshape(-1, 1))
        # WBC * Temp
        if wbc_idx is not None and temp_idx is not None:
            lab_interactions.append(
                (base[:, wbc_idx] * base[:, temp_idx]).reshape(-1, 1))
        # Creatinine * MAP (negative = bad)
        if creat_idx is not None and map_idx is not None:
            lab_interactions.append(
                (base[:, creat_idx] * base[:, map_idx]).reshape(-1, 1))
        # BUN * Lactate
        if bun_idx is not None and lactate_idx is not None:
            lab_interactions.append(
                (base[:, bun_idx] * base[:, lactate_idx]).reshape(-1, 1))
        # Platelets_diff * WBC_diff (trajectory interaction)
        plt_diff_name = "Platelets_diff"
        wbc_diff_name = "WBC_diff"
        # These diffs were computed earlier; use the differential features array
        clinical_cols = [c for c in feature_columns if c in VITAL_SIGNS + LAB_VALUES]
        if "Platelets" in clinical_cols and "WBC" in clinical_cols:
            plt_diff_idx = clinical_cols.index("Platelets")
            wbc_diff_idx = clinical_cols.index("WBC")
            lab_interactions.append(
                (diffs[:, plt_diff_idx] * diffs[:, wbc_diff_idx]).reshape(-1, 1))

        if lab_interactions:
            parts.extend(lab_interactions)

        # Lab deviation from population norm (absolute z-score)
        key_labs = ["Lactate", "WBC", "Creatinine", "BUN",
                    "Platelets", "Bilirubin_total", "Glucose", "Temp"]
        lab_dev_parts = []
        for lab in key_labs:
            if lab in feature_columns:
                idx = feature_columns.index(lab)
                lab_dev_parts.append(np.abs(base[:, idx:idx + 1]))
        if lab_dev_parts:
            parts.extend(lab_dev_parts)

    return np.hstack(parts)


def get_baseline_feature_names(feature_columns, enhanced=False, has_norm_stats=False):
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

        # Differential features for all clinical vars
        clinical_cols = [c for c in feature_columns if c in VITAL_SIGNS + LAB_VALUES]
        names.extend([f"{c}_diff" for c in clinical_cols])

        if has_norm_stats:
            # SOFA components (7)
            names.extend([
                "sofa_map", "sofa_creatinine", "sofa_bilirubin",
                "sofa_platelets", "sofa_respiration", "sofa_coagulation",
                "sofa_total",
            ])

            # qSOFA (3)
            names.extend(["qsofa_sbp", "qsofa_resp", "qsofa_total"])

            # NEWS scores (4)
            names.extend(["news_hr", "news_temp", "news_resp", "news_total"])

            # Clinical ratios (up to 5)
            if "HR" in feature_columns and "SBP" in feature_columns:
                names.append("shock_index")
            if "BUN" in feature_columns and "Creatinine" in feature_columns:
                names.append("bun_creatinine_ratio")
            if "SaO2" in feature_columns and "FiO2" in feature_columns:
                names.append("sao2_fio2_ratio")
            if "SBP" in feature_columns and "DBP" in feature_columns:
                names.append("pulse_pressure")
            if "SBP" in feature_columns and "DBP" in feature_columns and "HR" in feature_columns:
                names.append("cardiac_output_proxy")

            # SIRS score (5)
            names.extend([
                "sirs_temp", "sirs_hr", "sirs_resp", "sirs_wbc", "sirs_total",
            ])

            # MEWS score (5)
            names.extend([
                "mews_sbp", "mews_hr", "mews_resp", "mews_temp", "mews_total",
            ])

        # Multi-window rate of change (3h, 12h)
        names.extend([f"{c}_roc_3h" for c in vital_cols])
        names.extend([f"{c}_roc_12h" for c in vital_cols])

        # Second EMA half-life (6h)
        names.extend([f"{c}_tw_avg_6h" for c in vital_cols])

        # Lab cross-interactions
        if "Lactate" in feature_columns and "HR" in feature_columns:
            names.append("lactate_hr_product")
        if "WBC" in feature_columns and "Temp" in feature_columns:
            names.append("wbc_temp_product")
        if "Creatinine" in feature_columns and "MAP" in feature_columns:
            names.append("creatinine_map_product")
        if "BUN" in feature_columns and "Lactate" in feature_columns:
            names.append("bun_lactate_product")
        clinical_cols = [c for c in feature_columns if c in VITAL_SIGNS + LAB_VALUES]
        if "Platelets" in clinical_cols and "WBC" in clinical_cols:
            names.append("platelets_diff_wbc_diff_product")

        # Lab deviation from norm
        key_labs = ["Lactate", "WBC", "Creatinine", "BUN",
                    "Platelets", "Bilirubin_total", "Glucose", "Temp"]
        for lab in key_labs:
            if lab in feature_columns:
                names.append(f"{lab}_abs_zscore")

    return names


def prepare_baseline_data(patients_dict, feature_columns, enhanced=False, norm_stats=None):
    """Flatten patient dict into arrays for LightGBM.

    Args:
        patients_dict: Dict of preprocessed DataFrames (120 features + SepsisLabel).
        feature_columns: List of the 120 feature column names.
        enhanced: If True, compute additional engineered features.
        norm_stats: Dict of {feature: {mean, std, p1, p99}} for un-normalizing.
            Pass this to enable clinical score features (SOFA, qSOFA, NEWS, ratios).

    Returns:
        (X, y, patient_ids, hour_indices) where X is (n_samples, n_features).
    """
    all_X, all_y, all_pids, all_hours = [], [], [], []

    for pid, df in patients_dict.items():
        X_patient = create_baseline_features(df, feature_columns, enhanced=enhanced, norm_stats=norm_stats)
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

    def __init__(self, params=None, improved=False, scale_pos_weight=None):
        self.params = dict(self.IMPROVED_PARAMS if improved else self.DEFAULT_PARAMS)
        if scale_pos_weight is not None:
            self.params.pop("is_unbalance", None)
            self.params["scale_pos_weight"] = scale_pos_weight
        if params:
            self.params.update(params)
        self.model = None
        self.feature_names = None
        self.enhanced = improved

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=None,
            early_stopping_rounds=None, feature_names=None):
        """Train the model with optional early stopping on validation set.

        For improved/enhanced mode, early stopping is disabled and a fixed
        number of trees (500) are trained. Validation loss is monitored
        for logging only.
        """
        if num_boost_round is None:
            num_boost_round = 500 if self.enhanced else 1000
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
                                improved=False, scale_pos_weight=None):
    """End-to-end baseline training and evaluation.

    Args:
        train_patients: Dict of preprocessed training patient DataFrames.
        val_patients: Dict of preprocessed validation patient DataFrames.
        feature_columns: List of 120 feature column names.
        improved: If True, use enhanced features and improved hyperparameters.
        scale_pos_weight: If set, use explicit class weight instead of is_unbalance.

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
    spw_str = f", scale_pos_weight={scale_pos_weight}" if scale_pos_weight else ""
    print(f"Training LightGBM ({mode} params{spw_str})...")
    model = LightGBMBaseline(improved=improved, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

    print("Finding optimal thresholds...")
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
