"""Patient-level F1 metric utilities for sepsis prediction.

Aggregates timestep-level predictions to patient-level using max probability,
then computes classification metrics at the patient level.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate_patient_level(probs, labels, pids):
    """Aggregate timestep predictions to patient level using max probability.

    For each patient:
    - max_prob: highest predicted probability across all timesteps
    - has_sepsis: True if ANY timestep label is 1

    Args:
        probs: array-like of predicted probabilities (per timestep)
        labels: array-like of ground truth labels (per timestep)
        pids: list of patient IDs (per timestep)

    Returns:
        (patient_probs, patient_labels) as numpy arrays, sorted by patient ID
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=float)

    patient_data = defaultdict(lambda: {"has_sepsis": False, "max_prob": 0.0})
    for i in range(len(pids)):
        pid = pids[i]
        if labels_arr[i] == 1:
            patient_data[pid]["has_sepsis"] = True
        patient_data[pid]["max_prob"] = max(patient_data[pid]["max_prob"], float(probs_arr[i]))

    patient_ids = sorted(patient_data.keys())
    patient_probs = np.array([patient_data[p]["max_prob"] for p in patient_ids])
    patient_labels = np.array([1 if patient_data[p]["has_sepsis"] else 0 for p in patient_ids])
    return patient_probs, patient_labels


def compute_patient_metrics(probs, labels, pids, threshold):
    """Compute patient-level classification metrics at a given threshold.

    Args:
        probs: timestep-level predicted probabilities
        labels: timestep-level ground truth labels
        pids: timestep-level patient IDs
        threshold: decision threshold applied to patient max-probability

    Returns:
        dict with keys: p_f1, p_sensitivity, p_specificity, p_precision,
        p_accuracy, tp, fp, tn, fn, n_patients, n_sepsis, n_nonsepsis
    """
    patient_probs, patient_labels = aggregate_patient_level(probs, labels, pids)
    patient_preds = (patient_probs >= threshold).astype(int)

    tp = int(((patient_preds == 1) & (patient_labels == 1)).sum())
    fp = int(((patient_preds == 1) & (patient_labels == 0)).sum())
    tn = int(((patient_preds == 0) & (patient_labels == 0)).sum())
    fn = int(((patient_preds == 0) & (patient_labels == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "p_f1": f1,
        "p_sensitivity": sensitivity,
        "p_specificity": specificity,
        "p_precision": precision,
        "p_accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_patients": len(patient_labels),
        "n_sepsis": int(patient_labels.sum()),
        "n_nonsepsis": int((patient_labels == 0).sum()),
    }


def find_best_patient_f1_threshold(probs, labels, pids, thresholds=None):
    """Grid search for threshold that maximizes patient-level F1.

    Args:
        probs: timestep-level predicted probabilities
        labels: timestep-level ground truth labels
        pids: timestep-level patient IDs
        thresholds: array of thresholds to try (default: 0.05 to 0.95, step 0.01)

    Returns:
        (best_threshold, best_f1, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    patient_probs, patient_labels = aggregate_patient_level(probs, labels, pids)

    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = None

    for t in thresholds:
        preds = (patient_probs >= t).astype(int)

        tp = int(((preds == 1) & (patient_labels == 1)).sum())
        fp = int(((preds == 1) & (patient_labels == 0)).sum())
        tn = int(((preds == 0) & (patient_labels == 0)).sum())
        fn = int(((preds == 0) & (patient_labels == 1)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {
                "p_f1": f1,
                "p_sensitivity": sensitivity,
                "p_specificity": specificity,
                "p_precision": precision,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

    if best_metrics is None:
        best_metrics = compute_patient_metrics(probs, labels, pids, best_threshold)

    return best_threshold, best_f1, best_metrics


def compute_patient_auroc_auprc(probs, labels, pids):
    """Compute patient-level AUROC and AUPRC.

    Args:
        probs: timestep-level predicted probabilities
        labels: timestep-level ground truth labels
        pids: timestep-level patient IDs

    Returns:
        (auroc, auprc)
    """
    patient_probs, patient_labels = aggregate_patient_level(probs, labels, pids)

    if len(set(patient_labels)) < 2:
        return 0.0, 0.0

    auroc = roc_auc_score(patient_labels, patient_probs)
    auprc = average_precision_score(patient_labels, patient_probs)
    return auroc, auprc
