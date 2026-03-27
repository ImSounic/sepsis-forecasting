"""Evaluation script for patient-level F1 optimized models.

Loads models from outputs/models/patient_f1/, computes patient-level metrics,
evaluates ensembles, and saves results to outputs/patient_f1_comparison.csv.

Does NOT modify existing results in outputs/balanced_comparison.csv.

Usage:
    python -m src.training.evaluate_patient_f1 --config configs/optimized.yaml
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.data.preprocessing import TARGET
from src.models.ensemble import SepsisEnsemble
from src.models.gru import GRUWithAttention
from src.models.lightgbm_model import LightGBMBaseline, prepare_baseline_data
from src.training.patient_f1_utils import (
    aggregate_patient_level,
    compute_patient_auroc_auprc,
    compute_patient_metrics,
    find_best_patient_f1_threshold,
)
from src.training.train import load_config
from src.training.trainer import compute_utility_score


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


def evaluate_model(name, probs, labels, pids, hours):
    """Evaluate a model at its optimal patient-F1 threshold.

    Returns dict with all metrics.
    """
    # Patient-level F1 optimized threshold
    threshold, pf1, metrics = find_best_patient_f1_threshold(probs, labels, pids)

    # Patient-level AUROC/AUPRC
    p_auroc, p_auprc = compute_patient_auroc_auprc(probs, labels, pids)

    # Utility at this threshold (for reference)
    binary = (np.asarray(probs) >= threshold).astype(int).tolist()
    utility = compute_utility_score(
        np.asarray(labels).tolist(), binary, pids, hours,
    )

    # Timestep-level F1 at this threshold (for reference)
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels)
    binary_arr = (probs_arr >= threshold).astype(int)
    ts_tp = int(((binary_arr == 1) & (labels_arr == 1)).sum())
    ts_fp = int(((binary_arr == 1) & (labels_arr == 0)).sum())
    ts_fn = int(((binary_arr == 0) & (labels_arr == 1)).sum())
    ts_prec = ts_tp / (ts_tp + ts_fp) if (ts_tp + ts_fp) > 0 else 0.0
    ts_sens = ts_tp / (ts_tp + ts_fn) if (ts_tp + ts_fn) > 0 else 0.0
    ts_f1 = 2 * ts_prec * ts_sens / (ts_prec + ts_sens) if (ts_prec + ts_sens) > 0 else 0.0

    return {
        "model": name,
        "threshold": threshold,
        "p_f1": metrics["p_f1"],
        "p_sensitivity": metrics["p_sensitivity"],
        "p_specificity": metrics["p_specificity"],
        "p_precision": metrics["p_precision"],
        "p_auroc": p_auroc,
        "p_auprc": p_auprc,
        "tp_patients": metrics["tp"],
        "fp_patients": metrics["fp"],
        "tn_patients": metrics["tn"],
        "fn_patients": metrics["fn"],
        "utility": utility,
        "ts_f1": ts_f1,
        "ts_sensitivity": ts_sens,
        "ts_precision": ts_prec,
    }


def discover_patient_f1_models(model_dir):
    """Find all models in the patient_f1 subdirectory."""
    models = []
    pf1_dir = os.path.join(model_dir, "patient_f1")

    # LightGBM models
    lgb_dir = os.path.join(pf1_dir, "lightgbm")
    for pkl_file in sorted(glob.glob(os.path.join(lgb_dir, "*.pkl"))):
        name = os.path.basename(pkl_file).replace(".pkl", "")
        if name == "best_pf1":
            continue  # Skip the copy, we'll evaluate the named ones
        models.append({
            "name": f"LightGBM ({name})",
            "type": "lightgbm",
            "path": pkl_file,
        })

    # GRU model
    gru_dir = os.path.join(pf1_dir, "gru")
    gru_path = os.path.join(gru_dir, "best.pt")
    if os.path.exists(gru_path):
        models.append({
            "name": "GRU (patient_f1)",
            "type": "gru",
            "path": gru_path,
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
    train_stats = cached.get("train_stats", {})
    norm_stats = train_stats.get("normalize", None)
    print(f"Loaded {len(val_processed)} validation patients")
    if norm_stats:
        print(f"  Loaded normalization stats for {len(norm_stats)} features")

    # Patient counts
    n_sepsis = sum(1 for df in val_processed.values() if df[TARGET].max() == 1)
    n_nonsepsis = len(val_processed) - n_sepsis
    print(f"  Sepsis patients: {n_sepsis} | Non-sepsis: {n_nonsepsis}")

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
        val_processed, feature_columns, enhanced=True, norm_stats=norm_stats,
    )

    # Discover models
    available_models = discover_patient_f1_models(model_dir)
    print(f"\nFound {len(available_models)} patient-F1 models:")
    for m in available_models:
        print(f"  {m['name']}")

    if not available_models:
        print("No patient-F1 models found. Run train_patient_f1 first.")
        return

    # Evaluate each model
    all_results = []
    model_predictions = {}

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

        model_predictions[model_info["name"]] = {
            "probs": probs,
            "labels": labels,
            "pids": pids,
            "hours": hours,
        }

        result = evaluate_model(model_info["name"], probs, labels, pids, hours)
        all_results.append(result)

        print(
            f"  pF1={result['p_f1']:.4f} pSens={result['p_sensitivity']:.4f} "
            f"pSpec={result['p_specificity']:.4f} pPrec={result['p_precision']:.4f} "
            f"thr={result['threshold']:.2f}"
        )
        print(
            f"  pAUROC={result['p_auroc']:.4f} pAUPRC={result['p_auprc']:.4f} "
            f"util={result['utility']:.4f} | "
            f"TP={result['tp_patients']} FP={result['fp_patients']} "
            f"TN={result['tn_patients']} FN={result['fn_patients']}"
        )

    # Evaluate ensembles if both GRU and LightGBM exist
    gru_preds = {k: v for k, v in model_predictions.items() if "GRU" in k}
    lgb_preds = {k: v for k, v in model_predictions.items() if "LightGBM" in k}

    if gru_preds and lgb_preds:
        print("\nEvaluating ensembles...")
        for gru_name, gru_data in gru_preds.items():
            for lgb_name, lgb_data in lgb_preds.items():
                # Max ensemble
                ens_probs = SepsisEnsemble.combine_max(gru_data["probs"], lgb_data["probs"])
                ens_name = f"Ensemble max ({gru_name} + {lgb_name})"
                result = evaluate_model(ens_name, ens_probs, lgb_data["labels"], lgb_data["pids"], lgb_data["hours"])
                all_results.append(result)

                # Weighted ensembles
                for gw in [0.3, 0.5, 0.7]:
                    ens = SepsisEnsemble(gru_weight=gw)
                    ens_probs = ens.combine(gru_data["probs"], lgb_data["probs"])
                    w_name = f"Ensemble {gw:.1f}/{1-gw:.1f} ({gru_name} + {lgb_name})"
                    result = evaluate_model(w_name, ens_probs, lgb_data["labels"], lgb_data["pids"], lgb_data["hours"])
                    all_results.append(result)

    # Write CSV
    output_path = os.path.join("outputs", "patient_f1_comparison.csv")
    os.makedirs("outputs", exist_ok=True)

    fieldnames = [
        "model", "threshold",
        "p_f1", "p_sensitivity", "p_specificity", "p_precision",
        "p_auroc", "p_auprc",
        "tp_patients", "fp_patients", "tn_patients", "fn_patients",
        "utility", "ts_f1", "ts_sensitivity", "ts_precision",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            rounded = {}
            for key in row:
                if isinstance(row[key], float):
                    rounded[key] = round(row[key], 4)
                else:
                    rounded[key] = row[key]
            writer.writerow(rounded)

    print(f"\nResults written to {output_path}")

    # Print summary tables
    print("\n" + "=" * 110)
    print("PATIENT-LEVEL F1 COMPARISON (sorted by pF1)")
    print("=" * 110)
    print(f"{'Model':<55s} {'pF1':>6s} {'pSens':>6s} {'pSpec':>6s} {'pPrec':>6s} {'pAUROC':>6s} {'Util':>6s} {'Thr':>5s}")
    for r in sorted(all_results, key=lambda x: -x["p_f1"]):
        print(
            f"{r['model']:<55s} "
            f"{r['p_f1']:>6.4f} {r['p_sensitivity']:>6.4f} {r['p_specificity']:>6.4f} "
            f"{r['p_precision']:>6.4f} {r['p_auroc']:>6.4f} {r['utility']:>6.4f} {r['threshold']:>5.2f}"
        )

    # Detailed patient breakdown for top model
    best = max(all_results, key=lambda x: x["p_f1"])
    print(f"\nBEST MODEL: {best['model']}")
    print(f"  Patient-level F1:          {best['p_f1']:.4f}")
    print(f"  Patient Sensitivity:       {best['p_sensitivity']:.4f} ({best['tp_patients']}/{best['tp_patients']+best['fn_patients']} sepsis patients caught)")
    print(f"  Patient Specificity:       {best['p_specificity']:.4f} ({best['tn_patients']}/{best['tn_patients']+best['fp_patients']} non-sepsis correct)")
    print(f"  Patient Precision:         {best['p_precision']:.4f} ({best['tp_patients']}/{best['tp_patients']+best['fp_patients']} flags are real)")
    print(f"  Patient AUROC:             {best['p_auroc']:.4f}")
    print(f"  Patient AUPRC:             {best['p_auprc']:.4f}")
    print(f"  PhysioNet Utility:         {best['utility']:.4f}")
    print(f"  Threshold:                 {best['threshold']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate patient-F1 optimized models",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
