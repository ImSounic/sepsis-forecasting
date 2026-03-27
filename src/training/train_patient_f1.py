"""Patient-level F1 optimized training pipeline.

Trains LightGBM (primary) and optionally GRU models, selecting the best
configuration by patient-level F1 score instead of PhysioNet utility.

Models are saved to outputs/models/patient_f1/ to avoid disturbing existing results.

Usage:
    python -m src.training.train_patient_f1 --config configs/optimized.yaml
    python -m src.training.train_patient_f1 --config configs/optimized.yaml --model lightgbm
    python -m src.training.train_patient_f1 --config configs/optimized.yaml --model gru
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import create_data_loaders
from src.data.preprocessing import TARGET
from src.models.gru import GRUWithAttention, count_parameters
from src.models.lightgbm_model import (
    LightGBMBaseline,
    get_baseline_feature_names,
    prepare_baseline_data,
)
from src.training.patient_f1_utils import (
    compute_patient_auroc_auprc,
    compute_patient_metrics,
    find_best_patient_f1_threshold,
)
from src.training.train import load_config, set_seed, setup_logging
from src.training.trainer import SepsisTrainer, compute_utility_score


# LightGBM configuration grid
LGBM_CONFIGS = [
    {"name": "spw_5", "scale_pos_weight": 5, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "spw_10", "scale_pos_weight": 10, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "spw_20", "scale_pos_weight": 20, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "spw_50", "scale_pos_weight": 50, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "spw_100", "scale_pos_weight": 100, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "spw_10_big", "scale_pos_weight": 10, "num_leaves": 127, "learning_rate": 0.03},
    {"name": "spw_20_big", "scale_pos_weight": 20, "num_leaves": 127, "learning_rate": 0.03},
    {"name": "unbalance", "is_unbalance": True, "num_leaves": 63, "learning_rate": 0.05},
    {"name": "unbalance_big", "is_unbalance": True, "num_leaves": 127, "learning_rate": 0.03},
]


def train_lgb_patient_f1(config, train_processed, val_processed, logger, norm_stats=None, use_smote=False):
    """Train LightGBM models and select by patient-level F1."""
    sample_df = next(iter(train_processed.values()))
    feature_columns = [c for c in sample_df.columns if c != TARGET]
    feature_names = get_baseline_feature_names(
        feature_columns, enhanced=True, has_norm_stats=norm_stats is not None,
    )

    logger.info("Preparing LightGBM training data...")
    X_train, y_train, _, _ = prepare_baseline_data(
        train_processed, feature_columns, enhanced=True, norm_stats=norm_stats,
    )
    logger.info("Train: %d samples, %d features", X_train.shape[0], X_train.shape[1])

    # --- SMOTE oversampling on training data only ---
    if use_smote:
        if not HAS_SMOTE:
            logger.warning(
                "SMOTE requested but imbalanced-learn is not installed. "
                "Install with: pip install imbalanced-learn"
            )
        else:
            pos_before = int(y_train.sum())
            neg_before = len(y_train) - pos_before
            logger.info(
                "Before SMOTE: %d samples (pos=%d [%.2f%%], neg=%d)",
                len(y_train), pos_before, 100.0 * pos_before / len(y_train), neg_before,
            )

            # Oversample to 10% positive (not 50%) to avoid OOM
            # 23k pos → ~130k pos, keeping 1.29M neg = ~1.42M total
            target_pos = int(neg_before * 0.10)
            if target_pos <= pos_before:
                target_pos = pos_before * 5  # at least 5x oversample
            smote = SMOTE(
                random_state=42,
                sampling_strategy={1: target_pos, 0: neg_before},
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)

            pos_after = int(y_train.sum())
            neg_after = len(y_train) - pos_after
            logger.info(
                "After SMOTE:  %d samples (pos=%d [%.2f%%], neg=%d)",
                len(y_train), pos_after, 100.0 * pos_after / len(y_train), neg_after,
            )

    logger.info("Preparing LightGBM validation data...")
    X_val, y_val, val_pids, val_hours = prepare_baseline_data(
        val_processed, feature_columns, enhanced=True, norm_stats=norm_stats,
    )
    logger.info("Val: %d samples", X_val.shape[0])

    lgb_dir = os.path.join(config["output"]["model_dir"], "patient_f1", "lightgbm")
    os.makedirs(lgb_dir, exist_ok=True)

    all_results = []
    best_overall_f1 = 0.0
    best_overall_name = None

    for cfg in LGBM_CONFIGS:
        name = cfg["name"]
        logger.info("=" * 60)
        logger.info("Training config: %s", name)

        # Build params, overriding LightGBMBaseline defaults
        extra_params = {
            "num_leaves": cfg["num_leaves"],
            "learning_rate": cfg["learning_rate"],
        }

        use_unbalance = cfg.get("is_unbalance", False)
        spw = cfg.get("scale_pos_weight", None)

        if use_unbalance:
            model = LightGBMBaseline(improved=True, params=extra_params)
        else:
            model = LightGBMBaseline(improved=True, scale_pos_weight=spw, params=extra_params)

        # Train with fixed rounds (no early stopping — logloss early stop
        # is too aggressive with class weights, killing training at 1-3 trees)
        model.fit(
            X_train, y_train, X_val, y_val,
            num_boost_round=500,
            early_stopping_rounds=None,
            feature_names=feature_names,
        )

        val_probs = model.predict_proba(X_val)

        # Find best patient-level F1 threshold
        threshold, pf1, metrics = find_best_patient_f1_threshold(
            val_probs, y_val, val_pids,
        )

        # Also compute patient-level AUROC/AUPRC
        p_auroc, p_auprc = compute_patient_auroc_auprc(val_probs, y_val, val_pids)

        # Compute utility at this threshold for reference
        binary = (val_probs >= threshold).astype(int).tolist()
        utility = compute_utility_score(y_val.tolist(), binary, val_pids, val_hours)

        result = {
            "name": name,
            "threshold": threshold,
            "p_f1": metrics["p_f1"],
            "p_sensitivity": metrics["p_sensitivity"],
            "p_specificity": metrics["p_specificity"],
            "p_precision": metrics["p_precision"],
            "p_auroc": p_auroc,
            "p_auprc": p_auprc,
            "utility": utility,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "n_trees": model.model.num_trees(),
            "config": cfg,
        }
        all_results.append(result)

        logger.info(
            "  pF1=%.4f pSens=%.4f pSpec=%.4f pPrec=%.4f thr=%.2f util=%.4f",
            metrics["p_f1"], metrics["p_sensitivity"], metrics["p_specificity"],
            metrics["p_precision"], threshold, utility,
        )
        logger.info(
            "  pAUROC=%.4f pAUPRC=%.4f | TP=%d FP=%d TN=%d FN=%d",
            p_auroc, p_auprc, metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
        )

        # Save model
        save_path = os.path.join(lgb_dir, f"{name}.pkl")
        model.save(save_path)

        # Track best
        if metrics["p_f1"] > best_overall_f1:
            best_overall_f1 = metrics["p_f1"]
            best_overall_name = name
            model.save(os.path.join(lgb_dir, "best_pf1.pkl"))

        # Log top features
        importance = model.feature_importance()
        for _, row in importance.head(5).iterrows():
            logger.info("    %s: %.1f", row["feature"], row["importance"])

        del model
        gc.collect()

    # Print summary table
    logger.info("")
    logger.info("=" * 90)
    logger.info("PATIENT-LEVEL F1 RESULTS (LightGBM)")
    logger.info("=" * 90)
    logger.info(
        "%-20s %6s %6s %6s %6s %6s %6s %5s",
        "Config", "pF1", "pSens", "pSpec", "pPrec", "pAUROC", "Util", "Thr",
    )
    for r in sorted(all_results, key=lambda x: -x["p_f1"]):
        logger.info(
            "%-20s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %5.2f",
            r["name"], r["p_f1"], r["p_sensitivity"], r["p_specificity"],
            r["p_precision"], r["p_auroc"], r["utility"], r["threshold"],
        )
    logger.info("")
    logger.info("BEST: %s (pF1=%.4f)", best_overall_name, best_overall_f1)

    # Save results JSON
    results_path = os.path.join(lgb_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return all_results


class PatientF1Trainer(SepsisTrainer):
    """GRU trainer that early-stops on patient-level F1 instead of utility."""

    def find_optimal_threshold(self, predictions, labels, patient_ids, hour_indices):
        """Override: maximize patient-level F1 instead of utility."""
        threshold, pf1, _ = find_best_patient_f1_threshold(
            predictions, labels, patient_ids,
        )
        return threshold, pf1

    def find_optimal_thresholds(self, predictions, labels, patient_ids, hour_indices):
        """Override: add patient_f1 metric alongside utility/f1/youden."""
        # Call parent for utility, timestep-f1, youden
        base = super().find_optimal_thresholds(predictions, labels, patient_ids, hour_indices)

        # Add patient-level F1
        threshold, pf1, _ = find_best_patient_f1_threshold(
            predictions, labels, patient_ids,
        )
        base["patient_f1"] = {"threshold": threshold, "score": pf1}

        return base

    def train(self, epochs, early_stop_metric="patient_f1", checkpoint_subdir="gru"):
        """Override default early_stop_metric to patient_f1."""
        return super().train(epochs, early_stop_metric=early_stop_metric, checkpoint_subdir=checkpoint_subdir)


def train_gru_patient_f1(config, train_processed, val_processed, logger):
    """Train GRU model optimizing for patient-level F1."""
    seed = config.get("seed", 42)
    set_seed(seed)

    seq_length = config["data"]["seq_length"]
    train_loader, val_loader = create_data_loaders(
        train_processed,
        val_processed,
        batch_size=config["training"]["batch_size"],
        seq_length=seq_length,
        num_workers=0,  # Windows paging file constraint
    )

    model = GRUWithAttention(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
        attention_size=config["model"].get("attention_size", 64),
    )
    total, trainable = count_parameters(model)
    logger.info("GRU Model: %s params (%s trainable)", f"{total:,}", f"{trainable:,}")

    checkpoint_subdir = "patient_f1/gru"

    trainer_config = {
        "device": config.get("device", "auto"),
        "pos_weight": "auto",
        "lr": config["training"]["lr"],
        "weight_decay": config["training"]["weight_decay"],
        "max_grad_norm": config["training"]["gradient_clip"],
        "mixed_precision": config["training"].get("mixed_precision", True),
        "patience": config["training"]["patience"],
        "scheduler_patience": config["training"].get("scheduler_patience", 5),
        "scheduler_factor": config["training"].get("scheduler_factor", 0.5),
        "min_lr": config["training"].get("min_lr", 1e-5),
        "checkpoint_dir": config["output"]["model_dir"],
        "log_dir": os.path.join(config["output"]["log_dir"], "gru_patient_f1"),
    }

    trainer = PatientF1Trainer(model, train_loader, val_loader, trainer_config)
    logger.info("Positive class weight: %.2f", trainer.pos_weight)
    logger.info("Starting GRU training (early_stop=patient_f1)...")

    history = trainer.train(
        config["training"]["epochs"],
        early_stop_metric="patient_f1",
        checkpoint_subdir=checkpoint_subdir,
    )

    if history.get("val_f1"):
        # Find best epoch by patient_f1 if tracked, else by val_f1
        pf1_key = "val_patient_f1" if "val_patient_f1" in history else "val_f1"
        if pf1_key in history and history[pf1_key]:
            best_idx = int(np.argmax(history[pf1_key]))
            logger.info(
                "GRU done. Best epoch %d: pF1=%.4f, Utility=%.4f",
                best_idx + 1,
                history[pf1_key][best_idx],
                history["val_utility"][best_idx],
            )

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(config_path, model_type="all", use_smote=False):
    config = load_config(config_path)
    logger = setup_logging(config["output"]["log_dir"])
    seed = config.get("seed", 42)
    set_seed(seed)

    # Load cached preprocessed data
    cache_path = get_cache_path(config)
    logger.info("Loading preprocessed data from cache...")
    cached = load_preprocessed(cache_path)
    train_processed = cached["train_processed"]
    val_processed = cached["val_processed"]
    train_stats = cached.get("train_stats", {})
    norm_stats = train_stats.get("normalize", None)
    logger.info("Loaded %d train, %d val patients", len(train_processed), len(val_processed))
    if norm_stats:
        logger.info("Loaded normalization stats for %d features", len(norm_stats))

    if model_type in ("all", "lightgbm"):
        train_lgb_patient_f1(config, train_processed, val_processed, logger, norm_stats=norm_stats, use_smote=use_smote)

    if model_type in ("all", "gru"):
        train_gru_patient_f1(config, train_processed, val_processed, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train sepsis models optimized for patient-level F1",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--model", choices=["all", "gru", "lightgbm"], default="all",
        help="Which model type to train (default: all)",
    )
    parser.add_argument(
        "--smote", action="store_true", default=False,
        help="Apply SMOTE oversampling to training data (default: False)",
    )
    parser.add_argument(
        "--no-smote", action="store_false", dest="smote",
        help="Disable SMOTE oversampling",
    )
    args = parser.parse_args()
    main(args.config, model_type=args.model, use_smote=args.smote)
