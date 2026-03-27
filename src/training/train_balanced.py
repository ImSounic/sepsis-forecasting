"""Batch training script for balanced-metric sepsis models.

Trains multiple GRU and LightGBM models with different class weight settings,
optimizing for F1 instead of PhysioNet utility to improve sensitivity.

Usage:
    python -m src.training.train_balanced --config configs/optimized.yaml
    python -m src.training.train_balanced --config configs/optimized.yaml --model gru
    python -m src.training.train_balanced --config configs/optimized.yaml --model lightgbm
"""

import argparse
import gc
import os

import numpy as np
import torch

from src.data.cache import get_cache_path, load_preprocessed
from src.data.dataset import create_data_loaders
from src.data.preprocessing import TARGET
from src.models.gru import GRUWithAttention, count_parameters
from src.models.lightgbm_model import (
    LightGBMBaseline,
    get_baseline_feature_names,
    prepare_baseline_data,
)
from src.training.train import load_config, set_seed, setup_logging
from src.training.trainer import SepsisTrainer


GRU_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
LGB_SCALE_POS_WEIGHTS = [1.0, 5.0, 10.0, 20.0, 53.0]


def train_gru_balanced(config, train_processed, val_processed, logger):
    """Train GRU models with different pos_weight_multipliers."""
    seed = config.get("seed", 42)
    seq_length = config["data"]["seq_length"]

    train_loader, val_loader = create_data_loaders(
        train_processed,
        val_processed,
        batch_size=config["training"]["batch_size"],
        seq_length=seq_length,
        num_workers=config["training"].get("num_workers", 4),
    )

    for mult in GRU_MULTIPLIERS:
        set_seed(seed)
        logger.info("Training GRU with pos_weight_multiplier=%.1f", mult)

        model = GRUWithAttention(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
            attention_size=config["model"].get("attention_size", 64),
        )
        total, trainable = count_parameters(model)
        logger.info("Model: %s params (%s trainable)", f"{total:,}", f"{trainable:,}")

        checkpoint_subdir = f"gru_balanced/pwm_{mult}"

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
            "log_dir": os.path.join(config["output"]["log_dir"], f"gru_balanced_pwm_{mult}"),
        }
        trainer = SepsisTrainer(model, train_loader, val_loader, trainer_config)

        # Apply multiplier
        trainer.pos_weight = trainer.pos_weight * mult
        logger.info("Effective pos_weight: %.2f", trainer.pos_weight)

        history = trainer.train(
            config["training"]["epochs"],
            early_stop_metric="f1",
            checkpoint_subdir=checkpoint_subdir,
        )

        if history["val_f1"]:
            best_idx = int(np.argmax(history["val_f1"]))
            logger.info(
                "pwm=%.1f done. Best epoch %d: F1=%.4f, Utility=%.4f, Youden=%.4f",
                mult, best_idx + 1,
                history["val_f1"][best_idx],
                history["val_utility"][best_idx],
                history["val_youden"][best_idx],
            )

        # Free GPU memory between runs
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All GRU balanced models trained.")


def train_lgb_balanced(config, train_processed, val_processed, logger):
    """Train LightGBM models with different scale_pos_weight values."""
    sample_df = next(iter(train_processed.values()))
    feature_columns = [c for c in sample_df.columns if c != TARGET]
    feature_names = get_baseline_feature_names(feature_columns, enhanced=True)

    logger.info("Preparing LightGBM training data...")
    X_train, y_train, _, _ = prepare_baseline_data(
        train_processed, feature_columns, enhanced=True,
    )
    logger.info("Train: %d samples, %d features", X_train.shape[0], X_train.shape[1])

    logger.info("Preparing LightGBM validation data...")
    X_val, y_val, val_pids, val_hours = prepare_baseline_data(
        val_processed, feature_columns, enhanced=True,
    )
    logger.info("Val: %d samples", X_val.shape[0])

    lgb_dir = os.path.join(config["output"]["model_dir"], "lightgbm_balanced")
    os.makedirs(lgb_dir, exist_ok=True)

    for spw in LGB_SCALE_POS_WEIGHTS:
        logger.info("Training LightGBM with scale_pos_weight=%.1f", spw)

        model = LightGBMBaseline(improved=True, scale_pos_weight=spw)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)

        val_probs = model.predict_proba(X_val)

        # Find best threshold by utility (for comparison)
        from src.training.trainer import compute_utility_score
        best_threshold, best_score = 0.5, float("-inf")
        for t in np.arange(0.1, 0.91, 0.05):
            binary = (val_probs >= t).astype(int).tolist()
            score = compute_utility_score(y_val.tolist(), binary, val_pids, val_hours)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        logger.info("spw=%.1f: Utility=%.4f, Threshold=%.2f", spw, best_score, best_threshold)

        save_path = os.path.join(lgb_dir, f"spw_{spw}.pkl")
        model.save(save_path)
        logger.info("Saved to %s", save_path)

        # Log top features
        importance = model.feature_importance()
        for _, row in importance.head(10).iterrows():
            logger.info("  %s: %.1f", row["feature"], row["importance"])

    logger.info("All LightGBM balanced models trained.")


def main(config_path, model_type="all"):
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
    logger.info("Loaded %d train, %d val patients", len(train_processed), len(val_processed))

    if model_type in ("all", "gru"):
        train_gru_balanced(config, train_processed, val_processed, logger)

    if model_type in ("all", "lightgbm"):
        train_lgb_balanced(config, train_processed, val_processed, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train balanced-metric sepsis models with varying class weights",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--model", choices=["all", "gru", "lightgbm"], default="all",
        help="Which model type to train (default: all)",
    )
    args = parser.parse_args()
    main(args.config, model_type=args.model)
