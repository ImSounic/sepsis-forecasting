"""Comprehensive evaluation of all models with CSV output.

Loads original and balanced models, evaluates each with utility/F1/Youden
thresholds, computes full metrics, and outputs everything to CSV.

Usage:
    python -m src.training.evaluate_balanced --config configs/optimized.yaml
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.data.preprocessing import TARGET
from src.models.ensemble import SepsisEnsemble
from src.models.gru import GRUWithAttention
from src.models.lightgbm_model import LightGBMBaseline, prepare_baseline_data
from src.training.train import load_config
from src.training.trainer import (
    compute_f1_score,
    compute_utility_score,
    compute_youden_j,
)


def get_gru_predictions(model, val_loader, device):
    """Run GRU inference. Returns (probs, labels, pids, hours)."""
    model.eval()
    all_probs, all_labels, all_pids, all_hours = [], [], [], []

    with torch.no_grad():
        for features, labels, masks, pids, hours in val_loader:
            features = features.to(device)
            masks = masks.to(device)
            probs, _ = model.predict_proba(features, padding_mask=masks)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_pids.extend(list(pids))
            all_hours.extend([int(h) for h in hours])

    return np.array(all_probs), np.array(all_labels), all_pids, all_hours


def _aggregate_patient_level(probs_arr, labels_arr, pids):
    """Aggregate timestep predictions to patient level using max probability.

    For each patient:
    - has_sepsis: True if ANY timestep label is 1
    - max_prob: highest predicted probability across all timesteps

    Returns (patient_max_probs, patient_labels) as parallel arrays.
    """
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


def _patient_level_f1(probs_arr, labels_arr, pids, threshold):
    """Compute patient-level F1 using max-probability aggregation."""
    patient_probs, patient_labels = _aggregate_patient_level(probs_arr, labels_arr, pids)
    patient_preds = (patient_probs >= threshold).astype(int)

    tp = int(((patient_preds == 1) & (patient_labels == 1)).sum())
    fp = int(((patient_preds == 1) & (patient_labels == 0)).sum())
    fn = int(((patient_preds == 0) & (patient_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def find_all_thresholds(probs, labels, pids, hours):
    """Find optimal thresholds for utility, F1 (timestep), patient-level F1, and Youden's J.

    Returns dict: {metric_name: {'threshold': float, 'score': float}}
    """
    thresholds = np.arange(0.05, 0.91, 0.01)
    labels_arr = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    probs_arr = np.array(probs) if not isinstance(probs, np.ndarray) else probs

    best = {
        "utility": {"threshold": 0.5, "score": float("-inf")},
        "f1": {"threshold": 0.5, "score": float("-inf")},
        "patient_f1": {"threshold": 0.5, "score": float("-inf")},
        "youden": {"threshold": 0.5, "score": float("-inf")},
    }

    for t in thresholds:
        binary = (probs_arr >= t).astype(int)
        binary_list = binary.tolist()

        util = compute_utility_score(labels_arr.tolist(), binary_list, pids, hours)
        if util > best["utility"]["score"]:
            best["utility"] = {"threshold": float(t), "score": util}

        f1 = compute_f1_score(labels_arr.tolist(), binary_list)
        if f1 > best["f1"]["score"]:
            best["f1"] = {"threshold": float(t), "score": f1}

        pf1 = _patient_level_f1(probs_arr, labels_arr, pids, float(t))
        if pf1 > best["patient_f1"]["score"]:
            best["patient_f1"] = {"threshold": float(t), "score": pf1}

        youden = compute_youden_j(labels_arr.tolist(), binary_list)
        if youden > best["youden"]["score"]:
            best["youden"] = {"threshold": float(t), "score": youden}

    return best


def compute_full_metrics(probs, labels, pids, hours, threshold, val_processed=None):
    """Compute comprehensive metrics at a given threshold.

    Returns dict with all metrics including patient-level counts.
    """
    preds_arr = np.array(probs)
    labels_arr = np.array(labels)
    binary = (preds_arr >= threshold).astype(int)

    tp = int(((binary == 1) & (labels_arr == 1)).sum())
    fp = int(((binary == 1) & (labels_arr == 0)).sum())
    tn = int(((binary == 0) & (labels_arr == 0)).sum())
    fn = int(((binary == 0) & (labels_arr == 1)).sum())
    total = tp + fp + tn + fn

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    auroc = roc_auc_score(labels_arr, preds_arr) if len(set(labels_arr)) > 1 else 0.0
    auprc = average_precision_score(labels_arr, preds_arr) if len(set(labels_arr)) > 1 else 0.0

    utility = compute_utility_score(
        labels_arr.tolist(), binary.tolist(), pids, hours,
    )

    # Patient-level metrics using max-probability aggregation
    # Each patient gets ONE score = max predicted probability across all timesteps
    # Then threshold is applied to that single score
    patient_probs, patient_labels = _aggregate_patient_level(preds_arr, labels_arr, pids)
    patient_preds = (patient_probs >= threshold).astype(int)

    patient_tp = int(((patient_preds == 1) & (patient_labels == 1)).sum())
    patient_fp = int(((patient_preds == 1) & (patient_labels == 0)).sum())
    patient_tn = int(((patient_preds == 0) & (patient_labels == 0)).sum())
    patient_fn = int(((patient_preds == 0) & (patient_labels == 1)).sum())

    p_total = patient_tp + patient_fp + patient_tn + patient_fn
    p_sensitivity = patient_tp / (patient_tp + patient_fn) if (patient_tp + patient_fn) > 0 else 0.0
    p_specificity = patient_tn / (patient_tn + patient_fp) if (patient_tn + patient_fp) > 0 else 0.0
    p_precision = patient_tp / (patient_tp + patient_fp) if (patient_tp + patient_fp) > 0 else 0.0
    p_f1 = 2 * p_precision * p_sensitivity / (p_precision + p_sensitivity) if (p_precision + p_sensitivity) > 0 else 0.0
    p_accuracy = (patient_tp + patient_tn) / p_total if p_total > 0 else 0.0

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "utility": utility,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tp_patients": patient_tp,
        "fp_patients": patient_fp,
        "tn_patients": patient_tn,
        "fn_patients": patient_fn,
        "p_sensitivity": p_sensitivity,
        "p_specificity": p_specificity,
        "p_precision": p_precision,
        "p_f1": p_f1,
        "p_accuracy": p_accuracy,
    }


def load_gru_model(checkpoint_path, config, device):
    """Load a GRU model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GRUWithAttention(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        attention_size=config["model"].get("attention_size", 64),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def discover_models(model_dir):
    """Find all available model checkpoints.

    Returns list of dicts: [{'name': str, 'type': 'gru'|'lightgbm', 'path': str, 'config_info': str}]
    """
    models = []

    # Original GRU
    gru_dir = os.path.join(model_dir, "gru")
    for pt_file in sorted(glob.glob(os.path.join(gru_dir, "*.pt"))):
        models.append({
            "name": "GRU (original)",
            "type": "gru",
            "path": pt_file,
            "config_info": "pwm=0.5 (utility-optimized)",
        })

    # Balanced GRU models
    balanced_gru_dir = os.path.join(model_dir, "gru_balanced")
    for pwm_dir in sorted(glob.glob(os.path.join(balanced_gru_dir, "pwm_*"))):
        pt_file = os.path.join(pwm_dir, "best.pt")
        if os.path.exists(pt_file):
            pwm_val = os.path.basename(pwm_dir).replace("pwm_", "")
            models.append({
                "name": f"GRU (pwm={pwm_val})",
                "type": "gru",
                "path": pt_file,
                "config_info": f"pwm={pwm_val} (f1-optimized)",
            })

    # Original LightGBM
    lgb_dir = os.path.join(model_dir, "lightgbm")
    improved_path = os.path.join(lgb_dir, "lightgbm_improved.pkl")
    baseline_path = os.path.join(lgb_dir, "lightgbm_baseline.pkl")
    if os.path.exists(improved_path):
        models.append({
            "name": "LightGBM (original improved)",
            "type": "lightgbm",
            "path": improved_path,
            "config_info": "is_unbalance=True",
        })
    elif os.path.exists(baseline_path):
        models.append({
            "name": "LightGBM (original baseline)",
            "type": "lightgbm",
            "path": baseline_path,
            "config_info": "is_unbalance=True",
        })

    # Balanced LightGBM models
    balanced_lgb_dir = os.path.join(model_dir, "lightgbm_balanced")
    for pkl_file in sorted(glob.glob(os.path.join(balanced_lgb_dir, "spw_*.pkl"))):
        spw_val = os.path.basename(pkl_file).replace("spw_", "").replace(".pkl", "")
        models.append({
            "name": f"LightGBM (spw={spw_val})",
            "type": "lightgbm",
            "path": pkl_file,
            "config_info": f"scale_pos_weight={spw_val}",
        })

    return models


def main(config_path):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = config["output"]["model_dir"]

    print(f"Device: {device}")

    # Load cached data
    cache_path = get_cache_path(config)
    cached = load_preprocessed(cache_path)
    val_processed = cached["val_processed"]
    print(f"Loaded {len(val_processed)} validation patients")

    # Create val loader for GRU
    seq_length = config["data"]["seq_length"]
    val_dataset = SepsisDataset(val_processed, seq_length=seq_length)
    feature_columns = val_dataset.feature_columns
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=0,
    )

    # Prepare LightGBM features once
    print("Preparing LightGBM features...")
    X_lgb, y_lgb, lgb_pids, lgb_hours = prepare_baseline_data(
        val_processed, feature_columns, enhanced=True,
    )

    # Discover all models
    available_models = discover_models(model_dir)
    print(f"\nFound {len(available_models)} models:")
    for m in available_models:
        print(f"  {m['name']} ({m['config_info']})")

    if not available_models:
        print("No models found. Run training first.")
        return

    # Evaluate each model
    all_results = []
    model_predictions = {}  # Store predictions for ensemble evaluation

    for model_info in available_models:
        print(f"\nEvaluating: {model_info['name']}...")

        if model_info["type"] == "gru":
            model, checkpoint = load_gru_model(model_info["path"], config, device)
            probs, labels, pids, hours = get_gru_predictions(model, val_loader, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            lgb_model = LightGBMBaseline.load(model_info["path"])
            probs = lgb_model.predict_proba(X_lgb)
            labels = y_lgb
            pids = lgb_pids
            hours = lgb_hours

        model_predictions[model_info["name"]] = probs

        # Find optimal thresholds for all three metrics
        thresholds = find_all_thresholds(probs, labels, pids, hours)

        for method_name, thresh_info in thresholds.items():
            metrics = compute_full_metrics(
                probs, labels, pids, hours,
                thresh_info["threshold"], val_processed,
            )
            row = {
                "model_name": model_info["name"],
                "model_type": model_info["type"],
                "config_info": model_info["config_info"],
                "threshold_method": method_name,
                **metrics,
            }
            all_results.append(row)

            print(
                f"  {method_name:>12s}: thr={metrics['threshold']:.2f} "
                f"ts[sens={metrics['sensitivity']:.4f} f1={metrics['f1']:.4f}] "
                f"pt[sens={metrics['p_sensitivity']:.4f} f1={metrics['p_f1']:.4f} prec={metrics['p_precision']:.4f}] "
                f"util={metrics['utility']:.4f}"
            )

    # Evaluate ensembles between best GRU and best LightGBM variants
    print("\nEvaluating ensembles...")
    gru_models = {k: v for k, v in model_predictions.items() if "GRU" in k}
    lgb_models = {k: v for k, v in model_predictions.items() if "LightGBM" in k}

    # Use labels/pids/hours from the last GRU evaluation (all share same val set)
    # Re-get from val_loader to be safe
    if gru_models:
        sample_gru_name = list(gru_models.keys())[0]
    if lgb_models:
        sample_lgb_name = list(lgb_models.keys())[0]

    for gru_name, gru_probs in gru_models.items():
        for lgb_name, lgb_probs in lgb_models.items():
            # Max ensemble
            ens_name = f"Ensemble max ({gru_name} + {lgb_name})"
            ens_probs = SepsisEnsemble.combine_max(gru_probs, lgb_probs)

            thresholds = find_all_thresholds(ens_probs, labels, pids, hours)
            for method_name, thresh_info in thresholds.items():
                metrics = compute_full_metrics(
                    ens_probs, labels, pids, hours,
                    thresh_info["threshold"], val_processed,
                )
                row = {
                    "model_name": ens_name,
                    "model_type": "ensemble",
                    "config_info": f"max({gru_name}, {lgb_name})",
                    "threshold_method": method_name,
                    **metrics,
                }
                all_results.append(row)

            # Weighted ensembles
            for gw in [0.3, 0.5, 0.7]:
                ens = SepsisEnsemble(gru_weight=gw)
                ens_probs = ens.combine(gru_probs, lgb_probs)
                w_name = f"Ensemble {gw:.1f}/{1-gw:.1f} ({gru_name} + {lgb_name})"

                thresholds = find_all_thresholds(ens_probs, labels, pids, hours)
                for method_name, thresh_info in thresholds.items():
                    metrics = compute_full_metrics(
                        ens_probs, labels, pids, hours,
                        thresh_info["threshold"], val_processed,
                    )
                    row = {
                        "model_name": w_name,
                        "model_type": "ensemble",
                        "config_info": f"weighted {gw:.1f}/{1-gw:.1f}",
                        "threshold_method": method_name,
                        **metrics,
                    }
                    all_results.append(row)

    # Write CSV
    output_path = os.path.join("outputs", "balanced_comparison.csv")
    os.makedirs("outputs", exist_ok=True)

    fieldnames = [
        "model_name", "model_type", "config_info", "threshold_method",
        "threshold", "sensitivity", "specificity", "precision", "f1",
        "accuracy", "auroc", "auprc", "utility",
        "tp", "fp", "tn", "fn",
        "tp_patients", "fp_patients", "tn_patients", "fn_patients",
        "p_sensitivity", "p_specificity", "p_precision", "p_f1", "p_accuracy",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            # Round float values for readability
            for key in row:
                if isinstance(row[key], float):
                    row[key] = round(row[key], 4)
            writer.writerow(row)

    print(f"\nResults written to {output_path}")
    print(f"Total rows: {len(all_results)} ({len(available_models)} models x 3 metrics + ensembles)")

    # Print summary tables
    print("\n" + "=" * 120)
    print("TOP 10 BY PATIENT-LEVEL F1 (patient_f1-threshold):")
    print("=" * 120)
    pf1_rows = [r for r in all_results if r["threshold_method"] == "patient_f1"]
    pf1_rows.sort(key=lambda r: -r["p_f1"])
    print(f"{'Model':<55s} {'pF1':>6s} {'pSens':>6s} {'pSpec':>6s} {'pPrec':>6s} {'Util':>6s} {'Thr':>5s}")
    for r in pf1_rows[:10]:
        print(
            f"{r['model_name']:<55s} "
            f"{r['p_f1']:>6.4f} {r['p_sensitivity']:>6.4f} {r['p_specificity']:>6.4f} "
            f"{r['p_precision']:>6.4f} {r['utility']:>6.4f} {r['threshold']:>5.2f}"
        )

    print(f"\n{'TOP 10 BY PATIENT-LEVEL F1 (utility-threshold):'}")
    util_rows = [r for r in all_results if r["threshold_method"] == "utility"]
    util_rows.sort(key=lambda r: -r["p_f1"])
    print(f"{'Model':<55s} {'pF1':>6s} {'pSens':>6s} {'pSpec':>6s} {'pPrec':>6s} {'Util':>6s} {'Thr':>5s}")
    for r in util_rows[:10]:
        print(
            f"{r['model_name']:<55s} "
            f"{r['p_f1']:>6.4f} {r['p_sensitivity']:>6.4f} {r['p_specificity']:>6.4f} "
            f"{r['p_precision']:>6.4f} {r['utility']:>6.4f} {r['threshold']:>5.2f}"
        )

    print(f"\n{'TOP 10 BY PATIENT SENSITIVITY (patient_f1-threshold):'}")
    pf1_rows.sort(key=lambda r: -r["p_sensitivity"])
    print(f"{'Model':<55s} {'pF1':>6s} {'pSens':>6s} {'pSpec':>6s} {'pPrec':>6s} {'Util':>6s} {'Thr':>5s}")
    for r in pf1_rows[:10]:
        print(
            f"{r['model_name']:<55s} "
            f"{r['p_f1']:>6.4f} {r['p_sensitivity']:>6.4f} {r['p_specificity']:>6.4f} "
            f"{r['p_precision']:>6.4f} {r['utility']:>6.4f} {r['threshold']:>5.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all models and generate comparison CSV",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
