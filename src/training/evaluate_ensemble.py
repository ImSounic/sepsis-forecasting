"""Evaluate ensemble strategies and ICULOS filtering on validation set.

Compares GRU alone, LightGBM alone, weighted ensembles, and ICULOS filtering.

Usage:
    python -m src.training.evaluate_ensemble --config configs/optimized.yaml
    python -m src.training.evaluate_ensemble --config configs/optimized.yaml --full-analysis
"""

import argparse
import os
import pickle

import numpy as np
import torch

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.models.lightgbm_model import LightGBMBaseline, prepare_baseline_data
from src.models.ensemble import SepsisEnsemble, filter_by_iculos
from src.models.gru import GRUWithAttention
from src.training.train import load_config
from src.training.trainer import compute_utility_score


def get_gru_predictions(model, val_loader, device):
    """Run GRU inference on all validation samples.

    Returns:
        (probs, labels, pids, hours) all as lists, aligned with DataLoader order.
    """
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


def find_best_threshold(probs, labels, pids, hours, thresholds=None):
    """Search for threshold that maximizes utility score.

    Returns:
        (best_threshold, best_utility)
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    best_thresh, best_util = 0.5, float("-inf")
    for t in thresholds:
        binary = (probs >= t).astype(int).tolist()
        score = compute_utility_score(
            labels.tolist(), binary, pids, hours,
        )
        if score > best_util:
            best_util = score
            best_thresh = float(t)
    return best_thresh, best_util


def run_full_analysis(name, probs, labels, pids, hours, threshold,
                      val_processed, gru_probs, lgb_probs, model_dir):
    """Run detailed analysis on the best ensemble strategy."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    preds_arr = np.array(probs)
    labels_arr = np.array(labels)
    binary_preds = (preds_arr >= threshold).astype(int)

    tp_mask = (binary_preds == 1) & (labels_arr == 1)
    fp_mask = (binary_preds == 1) & (labels_arr == 0)
    tn_mask = (binary_preds == 0) & (labels_arr == 0)
    fn_mask = (binary_preds == 0) & (labels_arr == 1)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    tn = int(tn_mask.sum())
    fn = int(fn_mask.sum())
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    auroc = roc_auc_score(labels_arr, preds_arr)
    auprc = average_precision_score(labels_arr, preds_arr)

    print(f"\nFULL ANALYSIS: {name} (threshold={threshold:.2f})\n")

    # Classification metrics
    print(f"\n  Classification Metrics:")
    print(f"    TP: {tp:>8,}    FP: {fp:>8,}")
    print(f"    FN: {fn:>8,}    TN: {tn:>8,}")
    print(f"    Total: {total:,}")
    print(f"\n    Precision:    {precision:.4f}")
    print(f"    Recall:       {recall:.4f}")
    print(f"    Specificity:  {specificity:.4f}")
    print(f"    F1 Score:     {f1:.4f}")
    print(f"    Accuracy:     {accuracy:.4f}")
    print(f"    AUROC:        {auroc:.4f}")
    print(f"    AUPRC:        {auprc:.4f}")

    # Prediction timing analysis
    _print_timing_analysis(preds_arr, labels_arr, binary_preds, pids, hours,
                           threshold, val_processed)

    # Model agreement analysis
    _print_agreement_analysis(gru_probs, lgb_probs, labels_arr, threshold)

    # Save ensemble predictions
    ens_dir = os.path.join(model_dir, "ensemble")
    os.makedirs(ens_dir, exist_ok=True)
    save_path = os.path.join(ens_dir, "ensemble_predictions.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "strategy": name,
            "threshold": threshold,
            "probs": preds_arr,
            "labels": labels_arr,
            "pids": pids,
            "hours": hours,
            "gru_probs": gru_probs,
            "lgb_probs": lgb_probs,
        }, f)
    print(f"\n  Predictions saved to {save_path}")


def _print_timing_analysis(preds_arr, labels_arr, binary_preds, pids, hours,
                           threshold, val_processed):
    """Analyze prediction timing relative to sepsis onset."""
    # Group samples by patient
    patient_samples = {}
    for i in range(len(pids)):
        pid = pids[i]
        if pid not in patient_samples:
            patient_samples[pid] = []
        patient_samples[pid].append({
            "hour": hours[i],
            "pred": preds_arr[i],
            "label": int(labels_arr[i]),
            "binary_pred": int(binary_preds[i]),
        })

    # Find sepsis onset hour per patient
    onset_hours = {}
    for pid, df in val_processed.items():
        sepsis_rows = df.index[df["SepsisLabel"] == 1]
        if len(sepsis_rows) > 0:
            onset_hours[pid] = int(sepsis_rows[0])

    def get_iculos(pid, hour_idx):
        if "ICULOS" in val_processed[pid].columns:
            return int(val_processed[pid]["ICULOS"].iloc[hour_idx])
        return hour_idx + 1

    # TP: patients with at least one correct positive prediction during sepsis
    tp_patients = {}
    for pid, samples in patient_samples.items():
        tp_samps = [s for s in samples if s["binary_pred"] == 1 and s["label"] == 1]
        if tp_samps:
            tp_patients[pid] = tp_samps

    tp_lead_times = []
    for pid in tp_patients:
        if pid not in onset_hours:
            continue
        onset = onset_hours[pid]
        first_tp = min(s["hour"] for s in tp_patients[pid])
        tp_lead_times.append(first_tp - onset)

    # FN: sepsis patients never flagged
    sepsis_pids = {pid for pid, samps in patient_samples.items()
                   if any(s["label"] == 1 for s in samps)}
    fn_pids = sepsis_pids - set(tp_patients.keys())

    fn_max_preds = []
    for pid in fn_pids:
        all_p = [s["pred"] for s in patient_samples[pid]]
        fn_max_preds.append(max(all_p))

    # FP: non-sepsis patients with false alarms
    fp_preds, fp_iculos = [], []
    fp_patient_count = 0
    for pid, samples in patient_samples.items():
        if pid in sepsis_pids:
            continue
        fp_samps = [s for s in samples if s["binary_pred"] == 1]
        if fp_samps:
            fp_patient_count += 1
            for s in fp_samps:
                fp_preds.append(s["pred"])
                fp_iculos.append(get_iculos(pid, s["hour"]))
    fp_preds = np.array(fp_preds) if fp_preds else np.array([])
    fp_iculos = np.array(fp_iculos) if fp_iculos else np.array([])

    # TN: non-sepsis patients never flagged positive
    non_sepsis_pids = set(patient_samples.keys()) - sepsis_pids
    tn_patient_count = len(non_sepsis_pids) - fp_patient_count
    total_patients = len(patient_samples)

    print(f"\n  Prediction Timing Analysis:")

    # Patient-level summary
    n_tp = len(tp_lead_times)
    n_fn = len(fn_max_preds)
    print(f"\n  Patient-level summary ({total_patients:,} total):")
    print(f"    True Positives:   {n_tp:>6,} patients")
    print(f"    True Negatives:   {tn_patient_count:>6,} patients (never flagged)")
    print(f"    False Positives:  {fp_patient_count:>6,} patients")
    print(f"    False Negatives:  {n_fn:>6,} patients (missed sepsis)")

    # TP timing
    print(f"\n  True Positives (n={n_tp} patients):")
    if n_tp > 0:
        lt = np.array(tp_lead_times)
        before_12 = int((lt < -12).sum())
        early = int(((lt >= -12) & (lt < -6)).sum())
        optimal = int(((lt >= -6) & (lt <= 0)).sum())
        late = int(((lt > 0) & (lt <= 3)).sum())
        very_late = int((lt > 3).sum())
        print(f"    < -12h (very early):       {before_12/n_tp*100:5.1f}% (n={before_12})")
        print(f"    -12h to -6h (early):       {early/n_tp*100:5.1f}% (n={early})")
        print(f"    -6h to 0h (optimal):       {optimal/n_tp*100:5.1f}% (n={optimal})")
        print(f"    0h to +3h (late):          {late/n_tp*100:5.1f}% (n={late})")
        print(f"    > +3h (very late):         {very_late/n_tp*100:5.1f}% (n={very_late})")
        print(f"    Mean lead time:            {-np.mean(lt):.1f}h before onset")
        print(f"    Median lead time:          {-np.median(lt):.1f}h before onset")

    # FN analysis
    n_fn = len(fn_max_preds)
    print(f"\n  False Negatives (n={n_fn} patients - missed sepsis):")
    if n_fn > 0:
        fn_arr = np.array(fn_max_preds)
        close_05 = int((fn_arr >= threshold - 0.05).sum())
        close_10 = int((fn_arr >= threshold - 0.10).sum())
        never = int((fn_arr < 0.20).sum())
        print(f"    Max pred >= {threshold-.05:.2f}:          "
              f"{close_05/n_fn*100:5.1f}% (n={close_05}) - close misses")
        print(f"    Max pred >= {threshold-.10:.2f}:          "
              f"{close_10/n_fn*100:5.1f}% (n={close_10})")
        print(f"    Max pred < 0.20:           "
              f"{never/n_fn*100:5.1f}% (n={never}) - never suspected")
        print(f"    Mean max prediction:       {fn_arr.mean():.4f}")

    # FP analysis
    n_fp = len(fp_preds)
    print(f"\n  False Positives (n={fp_patient_count} patients, {n_fp} samples):")
    if n_fp > 0:
        high = int((fp_preds > 0.70).sum())
        medium = int(((fp_preds >= threshold) & (fp_preds <= 0.70)).sum())
        early_stay = int((fp_iculos < 24).sum())
        late_stay = int((fp_iculos > 72).sum())
        print(f"    High confidence (>0.70):   {high/n_fp*100:5.1f}% (n={high})")
        print(f"    Medium ({threshold:.2f}-0.70):       "
              f"{medium/n_fp*100:5.1f}% (n={medium})")
        print(f"    ICULOS < 24h:              "
              f"{early_stay/n_fp*100:5.1f}% (n={early_stay}) - early in stay")
        print(f"    ICULOS > 72h:              "
              f"{late_stay/n_fp*100:5.1f}% (n={late_stay}) - late in stay")
        print(f"    Mean ICULOS at FP:         {fp_iculos.mean():.1f} hours")


def _print_agreement_analysis(gru_probs, lgb_probs, labels_arr, threshold):
    """Analyze agreement between GRU and LightGBM predictions."""
    gru_pos = gru_probs >= threshold
    lgb_pos = lgb_probs >= threshold

    both_pos = gru_pos & lgb_pos
    both_neg = ~gru_pos & ~lgb_pos
    gru_only = gru_pos & ~lgb_pos
    lgb_only = ~gru_pos & lgb_pos

    n = len(labels_arr)
    pos_labels = labels_arr == 1

    print(f"\n  Model Agreement Analysis (threshold={threshold:.2f}):")
    print(f"    {'Category':<25s} {'Count':>8s} {'%':>7s}  {'Sepsis%':>7s}")

    for name, mask in [("Both positive", both_pos),
                       ("Both negative", both_neg),
                       ("GRU only positive", gru_only),
                       ("LGB only positive", lgb_only)]:
        count = int(mask.sum())
        pct = count / n * 100
        sepsis_in = int((mask & pos_labels).sum())
        sepsis_pct = sepsis_in / count * 100 if count > 0 else 0.0
        print(f"    {name:<25s} {count:>8,} {pct:>6.1f}%  {sepsis_pct:>6.1f}%")

    # Complementary insights
    both_pos_tp = int((both_pos & pos_labels).sum())
    gru_only_tp = int((gru_only & pos_labels).sum())
    lgb_only_tp = int((lgb_only & pos_labels).sum())
    total_tp = both_pos_tp + gru_only_tp + lgb_only_tp

    print(f"\n    True positives breakdown:")
    if total_tp > 0:
        print(f"      Both agree:      {both_pos_tp:>6,} ({both_pos_tp/total_tp*100:.1f}%)")
        print(f"      GRU-only finds:  {gru_only_tp:>6,} ({gru_only_tp/total_tp*100:.1f}%)")
        print(f"      LGB-only finds:  {lgb_only_tp:>6,} ({lgb_only_tp/total_tp*100:.1f}%)")
    else:
        print(f"      No true positives at this threshold")


def main(config_path, full_analysis=False):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load cached data
    cache_path = get_cache_path(config)
    cached = load_preprocessed(cache_path)
    val_processed = cached["val_processed"]
    print(f"Loaded {len(val_processed)} validation patients")

    seq_length = config["data"]["seq_length"]
    val_dataset = SepsisDataset(val_processed, seq_length=seq_length)
    feature_columns = val_dataset.feature_columns
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=0,
    )
    print(f"Validation samples: {len(val_dataset):,}")

    # Load GRU model
    gru_dir = os.path.join(config["output"]["model_dir"], "gru")
    checkpoint_path = os.path.join(gru_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        import glob
        pt_files = glob.glob(os.path.join(gru_dir, "*.pt"))
        if pt_files:
            checkpoint_path = pt_files[0]
        else:
            print(f"ERROR: No GRU checkpoint in {gru_dir}")
            return
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
    print(f"Loaded GRU checkpoint (epoch {checkpoint['epoch']+1}, "
          f"utility={checkpoint['utility']:.4f})")

    # GRU predictions
    print("\nRunning GRU inference...")
    gru_probs, labels, pids, hours = get_gru_predictions(model, val_loader, device)
    print(f"  {len(gru_probs):,} predictions collected")

    # Load LightGBM model (prefer improved, fall back to baseline)
    model_dir = config["output"]["model_dir"]
    lgb_dir = os.path.join(model_dir, "lightgbm")
    improved_path = os.path.join(lgb_dir, "lightgbm_improved.pkl")
    baseline_path = os.path.join(lgb_dir, "lightgbm_baseline.pkl")

    if os.path.exists(improved_path):
        lgb_model = LightGBMBaseline.load(improved_path)
        lgb_enhanced = lgb_model.enhanced
        print(f"Loaded improved LightGBM ({lgb_model.model.num_trees()} trees, "
              f"enhanced={lgb_enhanced})")
    elif os.path.exists(baseline_path):
        lgb_model = LightGBMBaseline.load(baseline_path)
        lgb_enhanced = lgb_model.enhanced
        print(f"Loaded LightGBM baseline ({lgb_model.model.num_trees()} trees, "
              f"enhanced={lgb_enhanced})")
    else:
        print(f"\nERROR: No LightGBM model found in {model_dir}")
        print("Run: python -m src.training.train_baseline --config configs/default.yaml")
        return

    # LightGBM predictions (same sample order as GRU since both iterate dict order)
    print("Preparing LightGBM features...")
    X_lgb, y_lgb, lgb_pids, lgb_hours = prepare_baseline_data(
        val_processed, feature_columns, enhanced=lgb_enhanced,
    )
    print(f"  LightGBM features: {X_lgb.shape}")

    # Verify alignment
    assert len(lgb_pids) == len(pids), (
        f"Sample count mismatch: GRU={len(pids)}, LGB={len(lgb_pids)}"
    )
    mismatches = sum(1 for a, b in zip(pids[:1000], lgb_pids[:1000]) if a != b)
    if mismatches > 0:
        print(f"  WARNING: {mismatches}/1000 patient ID mismatches in first 1000 samples")
    else:
        print("  Sample alignment verified (first 1000 match)")

    print("Running LightGBM inference...")
    lgb_probs = lgb_model.predict_proba(X_lgb)
    print(f"  {len(lgb_probs):,} predictions collected")

    hours_arr = np.array(hours)

    # Define all strategies to evaluate
    strategies = []

    # 1. GRU alone
    strategies.append(("GRU alone", gru_probs))

    # 2. GRU + ICULOS filter
    for thresh in [6, 12]:
        filtered = filter_by_iculos(gru_probs, hours, iculos_threshold=thresh)
        strategies.append((f"GRU + ICULOS>={thresh}", filtered))

    # 3. LightGBM alone
    strategies.append(("LightGBM alone", lgb_probs))

    # 4. Weighted ensembles
    for gw in [0.5, 0.6, 0.7]:
        ens = SepsisEnsemble(gru_weight=gw)
        combined = ens.combine(gru_probs, lgb_probs)
        strategies.append((f"Ensemble ({gw:.1f}/{1-gw:.1f})", combined))

    # 5. Max ensemble
    combined_max = SepsisEnsemble.combine_max(gru_probs, lgb_probs)
    strategies.append(("Ensemble (max)", combined_max))

    # 6. OR ensemble
    combined_or = SepsisEnsemble.combine_or(gru_probs, lgb_probs, 0.3, 0.3)
    strategies.append(("Ensemble (OR 0.3)", combined_or))

    # 7. Best ensemble + ICULOS filter (evaluated after finding best ensemble)
    for gw in [0.5, 0.6, 0.7]:
        ens = SepsisEnsemble(gru_weight=gw)
        combined = ens.combine(gru_probs, lgb_probs)
        filtered = filter_by_iculos(combined, hours, iculos_threshold=6)
        strategies.append((f"Ens ({gw:.1f}/{1-gw:.1f}) + ICULOS>=6", filtered))

    # Evaluate all strategies
    print("\nENSEMBLE & FILTERING EVALUATION\n")
    print(f"{'Model':<30s} {'Utility':>8s} {'Threshold':>10s} {'Notes'}")

    results = []
    for name, probs in strategies:
        thresh, util = find_best_threshold(probs, labels, pids, hours)
        results.append((name, util, thresh))
        note = ""
        if "ICULOS" in name:
            n_zeroed = int((probs == 0).sum())
            note = f"  {n_zeroed:,} preds zeroed"
        print(f"{name:<30s} {util:>8.4f} {thresh:>10.2f}{note}")

    # Find best strategy
    best = max(results, key=lambda x: x[1])
    print(f"\n{'BEST: ' + best[0]:<30s} {best[1]:>8.4f} {best[2]:>10.2f}")

    print("\nDETAILED COMPARISON (top strategies)\n")

    ranked = sorted(results, key=lambda x: -x[1])
    for rank, (name, util, thresh) in enumerate(ranked[:5], 1):
        print(f"\n  #{rank} {name}")
        print(f"     Utility: {util:.4f}, Threshold: {thresh:.2f}")

        idx = [i for i, (n, _) in enumerate(strategies) if n == name][0]
        probs_s = strategies[idx][1]
        binary = (probs_s >= thresh).astype(int)
        tp = int(((binary == 1) & (labels == 1)).sum())
        fp = int(((binary == 1) & (labels == 0)).sum())
        fn = int(((binary == 0) & (labels == 1)).sum())
        tn = int(((binary == 0) & (labels == 0)).sum())
        print(f"     TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")

    # Full analysis on best strategy
    if full_analysis:
        best_name, best_util, best_thresh = best
        best_idx = [i for i, (n, _) in enumerate(strategies) if n == best_name][0]
        best_probs = strategies[best_idx][1]

        run_full_analysis(
            best_name, best_probs, labels, pids, hours, best_thresh,
            val_processed, gru_probs, lgb_probs, model_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble and filtering strategies",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--full-analysis", action="store_true",
                        help="Run detailed analysis on best strategy")
    args = parser.parse_args()
    main(args.config, full_analysis=args.full_analysis)
