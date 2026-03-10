"""Evaluate ensemble strategies and ICULOS filtering on validation set.

Compares GRU alone, LightGBM alone, weighted ensembles, and ICULOS filtering.

Usage:
    python -m src.training.evaluate_ensemble --config configs/optimized.yaml
"""

import argparse
import os

import numpy as np
import torch

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import SepsisDataset
from src.models.baseline import LightGBMBaseline, prepare_baseline_data
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


def main(config_path):
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
    checkpoint_path = os.path.join(config["output"]["model_dir"], "best.pt")
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

    # Load LightGBM model
    baseline_path = os.path.join(config["output"]["model_dir"], "baseline.pkl")
    if not os.path.exists(baseline_path):
        print(f"\nERROR: No LightGBM model at {baseline_path}")
        print("Run: python -m src.training.train_baseline --config configs/default.yaml")
        return

    lgb_model = LightGBMBaseline.load(baseline_path)
    print(f"Loaded LightGBM baseline ({lgb_model.model.num_trees()} trees)")

    # LightGBM predictions (same sample order as GRU since both iterate dict order)
    print("Preparing LightGBM features...")
    X_lgb, y_lgb, lgb_pids, lgb_hours = prepare_baseline_data(
        val_processed, feature_columns,
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
    print("\n" + "=" * 75)
    print("ENSEMBLE & FILTERING EVALUATION")
    print("=" * 75)
    print(f"{'Model':<30s} {'Utility':>8s} {'Threshold':>10s} {'Notes'}")
    print("-" * 75)

    results = []
    for name, probs in strategies:
        thresh, util = find_best_threshold(probs, labels, pids, hours)
        results.append((name, util, thresh))
        # Add notes
        note = ""
        if "ICULOS" in name:
            n_zeroed = int((probs == 0).sum())
            note = f"  {n_zeroed:,} preds zeroed"
        print(f"{name:<30s} {util:>8.4f} {thresh:>10.2f}{note}")

    # Find best strategy
    best = max(results, key=lambda x: x[1])
    print("-" * 75)
    print(f"{'BEST: ' + best[0]:<30s} {best[1]:>8.4f} {best[2]:>10.2f}")

    # Detailed comparison of top strategies
    print("\n" + "=" * 75)
    print("DETAILED COMPARISON (top strategies)")
    print("=" * 75)

    # Sort by utility descending
    ranked = sorted(results, key=lambda x: -x[1])
    for rank, (name, util, thresh) in enumerate(ranked[:5], 1):
        print(f"\n  #{rank} {name}")
        print(f"     Utility: {util:.4f}, Threshold: {thresh:.2f}")

        # Compute sample-level stats at optimal threshold
        idx = [i for i, (n, _) in enumerate(strategies) if n == name][0]
        probs_s = strategies[idx][1]
        binary = (probs_s >= thresh).astype(int)
        tp = int(((binary == 1) & (labels == 1)).sum())
        fp = int(((binary == 1) & (labels == 0)).sum())
        fn = int(((binary == 0) & (labels == 1)).sum())
        tn = int(((binary == 0) & (labels == 0)).sum())
        print(f"     TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble and filtering strategies",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
