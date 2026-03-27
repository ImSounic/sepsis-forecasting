"""Full threshold sweep across all patient-F1 models with complete metrics.

Outputs a CSV with every model x threshold combination, plus ensembles.
"""
import csv
import glob
import os

import numpy as np
import torch

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.data.preprocessing import TARGET
from src.models.gru import GRUWithAttention
from src.models.lightgbm_model import LightGBMBaseline, prepare_baseline_data
from src.training.patient_f1_utils import (
    compute_patient_auroc_auprc,
    compute_patient_metrics,
)
from src.training.train import load_config
from src.training.trainer import compute_utility_score

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def full_metrics(probs, y, pids, hrs, threshold):
    """Compute all patient-level metrics at a given threshold."""
    m = compute_patient_metrics(probs, y, pids, threshold)
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    total = tp + fp + tn + fn

    acc = (tp + tn) / total if total > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    youden = m["p_sensitivity"] + m["p_specificity"] - 1

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    # F2 score (weights recall 2x more than precision)
    beta2 = 4  # beta=2 squared
    prec, sens = m["p_precision"], m["p_sensitivity"]
    f2 = (1 + beta2) * prec * sens / (beta2 * prec + sens) if (beta2 * prec + sens) > 0 else 0

    # PhysioNet utility
    binary = (probs >= threshold).astype(int).tolist()
    util = compute_utility_score(y.tolist(), binary, pids, hrs)

    return {
        "threshold": threshold,
        "p_f1": m["p_f1"],
        "p_f2": f2,
        "accuracy": acc,
        "sensitivity": m["p_sensitivity"],
        "specificity": m["p_specificity"],
        "precision": m["p_precision"],
        "npv": npv,
        "youden_j": youden,
        "mcc": mcc,
        "utility": util,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def sweep_model(name, probs, y, pids, hrs):
    """Run full threshold sweep for one model."""
    p_auroc, p_auprc = compute_patient_auroc_auprc(probs, y, pids)
    results = []
    for t in THRESHOLDS:
        row = full_metrics(probs, y, pids, hrs, t)
        row["model"] = name
        row["p_auroc"] = p_auroc
        row["p_auprc"] = p_auprc
        results.append(row)
    return results


def main():
    config = load_config("configs/optimized.yaml")
    cached = load_preprocessed(get_cache_path(config))
    val = cached["val_processed"]
    norm_stats = cached.get("train_stats", {}).get("normalize")

    sample_df = next(iter(val.values()))
    fcols = [c for c in sample_df.columns if c != TARGET]
    X, y, pids, hrs = prepare_baseline_data(val, fcols, enhanced=True, norm_stats=norm_stats)

    n_sepsis = sum(1 for df in val.values() if df[TARGET].max() == 1)
    n_nonsepsis = len(val) - n_sepsis
    print(f"Validation: {n_sepsis} sepsis | {n_nonsepsis} non-sepsis patients")

    # Load all LightGBM models
    lgb_dir = os.path.join(config["output"]["model_dir"], "patient_f1", "lightgbm")
    model_predictions = {}

    for pkl_file in sorted(glob.glob(os.path.join(lgb_dir, "*.pkl"))):
        name = os.path.basename(pkl_file).replace(".pkl", "")
        if name == "best_pf1":
            continue
        print(f"Loading {name}...")
        model = LightGBMBaseline.load(pkl_file)
        model_predictions[name] = model.predict_proba(X)

    # Sweep all individual models
    all_results = []
    for name, probs in model_predictions.items():
        print(f"Sweeping {name}...")
        all_results.extend(sweep_model(f"LightGBM ({name})", probs, y, pids, hrs))

    # Ensembles between LightGBM models (top 3 by AUROC)
    # Pick pairs of different spw configs for ensembles
    model_names = list(model_predictions.keys())
    ensemble_pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            ensemble_pairs.append((model_names[i], model_names[j]))

    print(f"\nSweeping {len(ensemble_pairs)} ensemble pairs...")
    for m1, m2 in ensemble_pairs:
        p1, p2 = model_predictions[m1], model_predictions[m2]

        # Max ensemble
        ens_probs = np.maximum(p1, p2)
        ens_name = f"Ensemble max ({m1} + {m2})"
        all_results.extend(sweep_model(ens_name, ens_probs, y, pids, hrs))

        # Average ensemble
        ens_probs = 0.5 * p1 + 0.5 * p2
        ens_name = f"Ensemble avg ({m1} + {m2})"
        all_results.extend(sweep_model(ens_name, ens_probs, y, pids, hrs))

    # Load GRU model if available
    gru_dir = os.path.join(config["output"]["model_dir"], "patient_f1", "gru")
    gru_checkpoint = os.path.join(gru_dir, "best.pt")
    gru_probs = None

    if os.path.exists(gru_checkpoint):
        print("\nLoading GRU (patient_f1)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_length = config["data"]["seq_length"]

        model = GRUWithAttention(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
            attention_size=config["model"].get("attention_size", 64),
        )
        checkpoint = torch.load(gru_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        val_dataset = SepsisDataset(val, seq_length=seq_length)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False, num_workers=0,
        )

        all_preds, all_labels, all_pids_gru, all_hrs_gru = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                lengths = batch["lengths"].to(device)
                logits = model(x, lengths).squeeze(-1)
                probs_batch = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs_batch.tolist())
                all_labels.extend(batch["labels"].numpy().tolist())
                all_pids_gru.extend(batch["patient_ids"])
                all_hrs_gru.extend(batch["hour_indices"].numpy().tolist())

        gru_probs = np.array(all_preds, dtype=np.float32)
        gru_y = np.array(all_labels, dtype=np.float32)
        gru_pids = list(all_pids_gru)
        gru_hrs = list(all_hrs_gru)

        print(f"  GRU predictions: {len(gru_probs)} samples")
        print("Sweeping GRU (patient_f1)...")
        all_results.extend(sweep_model("GRU (patient_f1)", gru_probs, gru_y, gru_pids, gru_hrs))

        # Cross-model ensembles: GRU + each LightGBM
        # Need to align predictions by (patient_id, hour) since GRU/LGB have different sample orders
        print("\nBuilding GRU+LGB cross-model ensembles...")

        # Build GRU lookup: (pid, hour) -> prob
        gru_lookup = {}
        for i in range(len(gru_probs)):
            key = (gru_pids[i], int(gru_hrs[i]))
            gru_lookup[key] = gru_probs[i]

        # For each LGB model, align and create ensembles
        for lgb_name, lgb_probs_full in model_predictions.items():
            # Align GRU probs to LGB sample order
            gru_aligned = np.zeros_like(lgb_probs_full)
            matched = 0
            for i in range(len(pids)):
                key = (pids[i], int(hrs[i]))
                if key in gru_lookup:
                    gru_aligned[i] = gru_lookup[key]
                    matched += 1

            if matched < len(pids) * 0.5:
                print(f"  Skipping {lgb_name}: only {matched}/{len(pids)} samples matched")
                continue

            # Weighted ensembles
            for gru_w, lgb_w in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
                ens_probs = gru_w * gru_aligned + lgb_w * lgb_probs_full
                ens_name = f"Ensemble {gru_w}/{lgb_w} (GRU + {lgb_name})"
                all_results.extend(sweep_model(ens_name, ens_probs, y, pids, hrs))

            # Max ensemble
            ens_probs = np.maximum(gru_aligned, lgb_probs_full)
            ens_name = f"Ensemble max (GRU + {lgb_name})"
            all_results.extend(sweep_model(ens_name, ens_probs, y, pids, hrs))
    else:
        print("\nNo GRU model found at", gru_checkpoint)

    # Write CSV
    output_path = os.path.join("outputs", "full_threshold_sweep.csv")
    os.makedirs("outputs", exist_ok=True)

    fieldnames = [
        "model", "threshold", "p_f1", "p_f2", "accuracy", "sensitivity",
        "specificity", "precision", "npv", "youden_j", "mcc", "utility",
        "p_auroc", "p_auprc", "tp", "fp", "tn", "fn",
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

    n_models = len(set(r["model"] for r in all_results))
    print(f"\nWrote {len(all_results)} rows ({n_models} models x {len(THRESHOLDS)} thresholds)")
    print(f"Saved to {output_path}")

    # Print top 10 by patient F1
    print("\n" + "=" * 120)
    print("TOP 15 BY PATIENT-LEVEL F1")
    print("=" * 120)
    header = (
        f"{'Model':<50s} {'Thr':>5s} {'pF1':>6s} {'pF2':>6s} {'Acc':>6s} "
        f"{'Sens':>6s} {'Spec':>6s} {'Prec':>6s} {'NPV':>6s} "
        f"{'MCC':>6s} {'AUROC':>6s} {'Util':>6s}"
    )
    print(header)
    for r in sorted(all_results, key=lambda x: -x["p_f1"])[:15]:
        print(
            f"{r['model']:<50s} {r['threshold']:5.2f} "
            f"{r['p_f1']:6.4f} {r['p_f2']:6.4f} {r['accuracy']:6.4f} "
            f"{r['sensitivity']:6.4f} {r['specificity']:6.4f} {r['precision']:6.4f} "
            f"{r['npv']:6.4f} {r['mcc']:6.4f} {r['p_auroc']:6.4f} {r['utility']:6.4f}"
        )

    # Print top 10 by sensitivity (with pF1 > 0.30)
    print("\n" + "=" * 120)
    print("TOP 15 BY SENSITIVITY (pF1 > 0.30)")
    print("=" * 120)
    print(header)
    filtered = [r for r in all_results if r["p_f1"] > 0.30]
    for r in sorted(filtered, key=lambda x: -x["sensitivity"])[:15]:
        print(
            f"{r['model']:<50s} {r['threshold']:5.2f} "
            f"{r['p_f1']:6.4f} {r['p_f2']:6.4f} {r['accuracy']:6.4f} "
            f"{r['sensitivity']:6.4f} {r['specificity']:6.4f} {r['precision']:6.4f} "
            f"{r['npv']:6.4f} {r['mcc']:6.4f} {r['p_auroc']:6.4f} {r['utility']:6.4f}"
        )

    # Print top 10 by MCC
    print("\n" + "=" * 120)
    print("TOP 15 BY MCC")
    print("=" * 120)
    print(header)
    for r in sorted(all_results, key=lambda x: -x["mcc"])[:15]:
        print(
            f"{r['model']:<50s} {r['threshold']:5.2f} "
            f"{r['p_f1']:6.4f} {r['p_f2']:6.4f} {r['accuracy']:6.4f} "
            f"{r['sensitivity']:6.4f} {r['specificity']:6.4f} {r['precision']:6.4f} "
            f"{r['npv']:6.4f} {r['mcc']:6.4f} {r['p_auroc']:6.4f} {r['utility']:6.4f}"
        )


if __name__ == "__main__":
    main()
