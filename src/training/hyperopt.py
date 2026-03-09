"""Hyperparameter optimization with Optuna for the GRU sepsis model.

Usage:
    python -m src.training.hyperopt --config configs/default.yaml --n_trials 50
"""

import argparse
import gc
import logging
import os
import pickle
import random

import numpy as np
import optuna
import torch
import yaml

from src.data.cache import get_cache_path, is_cache_valid, load_preprocessed
from src.data.dataset import create_data_loaders
from src.models.gru import GRUWithAttention
from src.training.trainer import SepsisTrainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached_data(config):
    """Load preprocessed data from cache. Raises if cache is invalid."""
    raw_dir = config["data"]["raw_dir"]
    cache_path = get_cache_path(config)

    if not is_cache_valid(cache_path, raw_dir):
        raise FileNotFoundError(
            f"No valid cache at {cache_path}. Run full training first: "
            "python -m src.training.train --config configs/default.yaml"
        )

    cached = load_preprocessed(cache_path)
    print(f"Loaded {len(cached['train_processed'])} train, "
          f"{len(cached['val_processed'])} val patients from cache")
    return cached


class _OptunaCallback:
    """Report validation utility to Optuna for pruning after each epoch."""

    def __init__(self, trial):
        self.trial = trial
        self.epoch = 0

    def __call__(self, utility):
        self.trial.report(utility, self.epoch)
        self.epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def create_objective(config, train_processed, val_processed):
    """Create the Optuna objective function closed over the data."""

    def objective(trial):
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        pos_weight_mult = trial.suggest_categorical("pos_weight_multiplier", [0.5, 1.0, 2.0])

        set_seed(config.get("seed", 42))

        # Create data loaders with trial batch size
        train_loader, val_loader = create_data_loaders(
            train_processed,
            val_processed,
            batch_size=batch_size,
            seq_length=config["data"]["seq_length"],
            num_workers=config["training"].get("num_workers", 4),
        )

        # Build model
        model = GRUWithAttention(
            input_size=config["model"]["input_size"],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            attention_size=config["model"].get("attention_size", 64),
        )

        # Trainer config: 20 epochs max, patience 5
        trainer_config = {
            "device": config.get("device", "auto"),
            "pos_weight": "auto",
            "lr": lr,
            "weight_decay": config["training"]["weight_decay"],
            "max_grad_norm": config["training"]["gradient_clip"],
            "mixed_precision": config["training"].get("mixed_precision", True),
            "patience": 5,
            "scheduler_patience": 3,
            "scheduler_factor": 0.5,
            "min_lr": 1e-5,
            "checkpoint_dir": os.path.join("outputs", "optuna_models", f"trial_{trial.number}"),
            "log_dir": os.path.join("outputs", "optuna_logs", f"trial_{trial.number}"),
        }
        trainer = SepsisTrainer(model, train_loader, val_loader, trainer_config)

        # Apply pos_weight multiplier
        trainer.pos_weight = trainer.pos_weight * pos_weight_mult

        # Train with pruning callback
        callback = _OptunaCallback(trial)
        best_utility = float("-inf")

        for epoch in range(1, 21):
            train_loss = trainer.train_epoch()
            val_loss, preds, labels, pids, hours = trainer.validate()
            threshold, utility = trainer.find_optimal_threshold(preds, labels, pids, hours)
            trainer.scheduler.step(utility)

            if utility > best_utility:
                best_utility = utility
                patience_counter = 0
            else:
                patience_counter = patience_counter + 1 if epoch > 1 else 0
                if patience_counter >= 5:
                    break

            # Report to Optuna for pruning
            try:
                callback(utility)
            except optuna.TrialPruned:
                # Clean up before raising
                del model, trainer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()
                raise

            print(f"  Trial {trial.number} | Epoch {epoch}/20 | "
                  f"Utility: {utility:.4f} | LR: {lr:.1e} | "
                  f"h={hidden_size} l={num_layers} d={dropout:.1f} bi={bidirectional}")

        del model, trainer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        return best_utility

    return objective


def save_best_config(study, base_config, output_path="configs/best.yaml"):
    """Write best trial parameters as a new YAML config."""
    best = study.best_params

    new_config = {
        "data": base_config["data"].copy(),
        "model": {
            "input_size": base_config["model"]["input_size"],
            "hidden_size": best["hidden_size"],
            "num_layers": best["num_layers"],
            "dropout": best["dropout"],
            "bidirectional": best["bidirectional"],
            "attention_size": base_config["model"].get("attention_size", 64),
        },
        "training": {
            "batch_size": best["batch_size"],
            "epochs": base_config["training"]["epochs"],
            "lr": round(best["lr"], 6),
            "weight_decay": base_config["training"]["weight_decay"],
            "patience": base_config["training"]["patience"],
            "gradient_clip": base_config["training"]["gradient_clip"],
            "pos_weight": "auto",
            "pos_weight_multiplier": best["pos_weight_multiplier"],
            "mixed_precision": base_config["training"].get("mixed_precision", True),
            "num_workers": base_config["training"].get("num_workers", 4),
            "scheduler_patience": base_config["training"].get("scheduler_patience", 5),
            "scheduler_factor": base_config["training"].get("scheduler_factor", 0.5),
            "min_lr": base_config["training"].get("min_lr", 1e-5),
        },
        "output": base_config["output"].copy(),
        "seed": base_config.get("seed", 42),
        "device": base_config.get("device", "auto"),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved to {output_path}")


def main(config_path: str, n_trials: int = 50):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    # Load cached data once (shared across all trials)
    cached = load_cached_data(config)
    train_processed = cached["train_processed"]
    val_processed = cached["val_processed"]

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.get("seed", 42)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
        study_name="sepsis_gru_hyperopt",
    )

    objective = create_objective(config, train_processed, val_processed)

    print(f"\nStarting Optuna optimization: {n_trials} trials")
    print(f"Search space:")
    print(f"  hidden_size:  [64, 128, 256]")
    print(f"  num_layers:   [1, 2, 3]")
    print(f"  dropout:      [0.1, 0.3, 0.5]")
    print(f"  lr:           [1e-4, 1e-2] (log)")
    print(f"  batch_size:   [32, 64, 128]")
    print(f"  bidirectional: [True, False]")
    print(f"  pos_weight_multiplier: [0.5, 1.0, 2.0]")
    print()

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save study
    os.makedirs("outputs", exist_ok=True)
    study_path = "outputs/optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to {study_path}")

    # Print results
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Completed trials: {len(study.trials)}")
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Pruned trials:    {pruned}")
    print(f"  Best utility:     {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, val in study.best_params.items():
        print(f"  {key:25s} {val}")

    # Save best config
    save_best_config(study, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--config", required=True, help="Base config YAML file")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args.config, n_trials=args.n_trials)
