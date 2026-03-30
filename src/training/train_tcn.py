"""Training script for TCN sepsis prediction model.

Usage:
    python -m src.training.train_tcn --config configs/tcn.yaml
    python -m src.training.train_tcn --config configs/tcn.yaml --max_patients 100
"""

import argparse

import numpy as np

from src.models.gru import count_parameters
from src.models.tcn import TCNWithAttention
from src.training.train import (
    compute_train_stats,
    create_data_loaders,
    get_cache_path,
    get_train_val_split,
    is_cache_valid,
    load_all_patients,
    load_config,
    load_preprocessed,
    preprocess_patients,
    save_preprocessed,
    set_seed,
    setup_logging,
    SepsisTrainer,
)

import gc
import logging
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def main(config_path: str, max_patients: int = None, no_cache: bool = False,
         pos_weight_multiplier: float = None, early_stop_metric: str = None,
         checkpoint_name: str = None, resume: bool = False):
    config = load_config(config_path)
    logger = setup_logging(config["output"]["log_dir"])
    seed = config.get("seed", 42)
    set_seed(seed)

    if max_patients is not None:
        config["data"]["max_patients"] = max_patients

    raw_dir = config["data"]["raw_dir"]
    cache_path = get_cache_path(config)

    # Try loading from cache
    if not no_cache and is_cache_valid(cache_path, raw_dir):
        logger.info("Valid cache found, loading preprocessed data from %s", cache_path)
        cached = load_preprocessed(cache_path)
        train_processed = cached["train_processed"]
        val_processed = cached["val_processed"]
        train_ids = cached["train_ids"]
        val_ids = cached["val_ids"]
        train_stats = cached["train_stats"]
        logger.info("Loaded %d train, %d val patients from cache", len(train_processed), len(val_processed))
    else:
        if no_cache:
            logger.info("Cache disabled via --no-cache, processing fresh")
        else:
            logger.info("No valid cache found, processing from raw data")

        max_p = config["data"].get("max_patients")

        logger.info("Loading raw patient data from %s ...", raw_dir)
        patients = {}
        for subdir in ["training_setA", "training_setB"]:
            path = os.path.join(raw_dir, subdir)
            if os.path.isdir(path):
                sub = load_all_patients(path, max_patients=max_p)
                patients.update(sub)
                logger.info("  %s: %d patients", subdir, len(sub))

        if not patients:
            logger.error("No patient data found in %s", raw_dir)
            return

        logger.info("Total patients: %d", len(patients))

        train_ids, val_ids = get_train_val_split(
            list(patients.keys()),
            val_ratio=config["data"]["val_split"],
            seed=seed,
        )
        train_raw = {pid: patients[pid] for pid in train_ids}
        val_raw = {pid: patients[pid] for pid in val_ids}
        logger.info("Split: %d train, %d val", len(train_raw), len(val_raw))

        logger.info("Computing training statistics...")
        train_stats = compute_train_stats(train_raw)

        logger.info("Preprocessing %d training patients...", len(train_raw))
        train_processed = preprocess_patients(train_raw, train_stats)
        logger.info("Preprocessing %d validation patients...", len(val_raw))
        val_processed = preprocess_patients(val_raw, train_stats)

        del patients, train_raw, val_raw
        gc.collect()

        save_preprocessed(train_processed, val_processed, train_ids, val_ids, train_stats, cache_path)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_processed,
        val_processed,
        batch_size=config["training"]["batch_size"],
        seq_length=config["data"]["seq_length"],
        num_workers=config["training"].get("num_workers", 4),
    )
    logger.info(
        "DataLoaders: %d train samples, %d val samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Build TCN model
    model = TCNWithAttention(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        kernel_size=config["model"].get("kernel_size", 3),
        attention_size=config["model"].get("attention_size", 64),
    )
    total, trainable = count_parameters(model)
    logger.info("TCN Model: %s params (%s trainable)", f"{total:,}", f"{trainable:,}")

    # Initialize trainer (reuse SepsisTrainer — it's model-agnostic)
    trainer_config = {
        "device": config.get("device", "auto"),
        "pos_weight": config["training"].get("pos_weight", "auto"),
        "lr": config["training"]["lr"],
        "weight_decay": config["training"]["weight_decay"],
        "max_grad_norm": config["training"]["gradient_clip"],
        "mixed_precision": config["training"].get("mixed_precision", True),
        "patience": config["training"]["patience"],
        "scheduler_patience": config["training"].get("scheduler_patience", 5),
        "scheduler_factor": config["training"].get("scheduler_factor", 0.5),
        "min_lr": config["training"].get("min_lr", 1e-5),
        "checkpoint_dir": config["output"]["model_dir"],
        "log_dir": config["output"]["log_dir"],
    }
    trainer = SepsisTrainer(model, train_loader, val_loader, trainer_config)

    multiplier = pos_weight_multiplier or config["training"].get("pos_weight_multiplier", 1.0)
    if multiplier != 1.0:
        trainer.pos_weight = trainer.pos_weight * multiplier
    logger.info("Positive class weight: %.2f (multiplier=%.2f)", trainer.pos_weight, multiplier)

    # Train
    early_metric = early_stop_metric or "utility"
    checkpoint_sub = checkpoint_name or "tcn"

    # Resume from last checkpoint if requested
    resume_epoch = 0
    resume_best = None
    resume_pat = 0
    if resume:
        last_ckpt = os.path.join(config["output"]["model_dir"], checkpoint_sub, "last.pt")
        if os.path.exists(last_ckpt):
            resume_epoch, resume_best, resume_pat = trainer.resume_from_checkpoint(last_ckpt)
            logger.info("Resuming from epoch %d", resume_epoch)
        else:
            logger.warning("--resume flag set but no last.pt found, starting fresh")

    logger.info(
        "Starting TCN training for %d epochs (early_stop=%s, checkpoint=%s)...",
        config["training"]["epochs"], early_metric, checkpoint_sub,
    )
    history = trainer.train(
        config["training"]["epochs"],
        early_stop_metric=early_metric,
        checkpoint_subdir=checkpoint_sub,
        resume_epoch=resume_epoch,
        resume_best_score=resume_best,
        resume_patience=resume_pat,
    )

    # Final report
    if history["val_utility"]:
        metric_key = f"val_{early_metric}" if early_metric != "utility" else "val_utility"
        best_idx = int(np.argmax(history[metric_key]))
        logger.info(
            "Done. Best epoch %d: utility=%.4f, f1=%.4f, youden=%.4f, threshold=%.2f",
            best_idx + 1,
            history["val_utility"][best_idx],
            history["val_f1"][best_idx],
            history["val_youden"][best_idx],
            history["threshold"][best_idx],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN sepsis prediction model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Limit patients loaded (per dataset directory) for quick debugging",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force reprocessing even if a valid cache exists",
    )
    parser.add_argument(
        "--pos-weight-multiplier", type=float, default=None,
        help="Override pos_weight_multiplier from config",
    )
    parser.add_argument(
        "--early-stop-metric", choices=["utility", "f1", "youden"], default=None,
        help="Metric for early stopping and checkpointing (default: utility)",
    )
    parser.add_argument(
        "--checkpoint-name", default=None,
        help="Subdirectory name under model_dir for checkpoints (default: tcn)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint (last.pt)",
    )
    args = parser.parse_args()
    main(
        args.config,
        max_patients=args.max_patients,
        no_cache=args.no_cache,
        pos_weight_multiplier=args.pos_weight_multiplier,
        early_stop_metric=args.early_stop_metric,
        checkpoint_name=args.checkpoint_name,
        resume=args.resume,
    )
