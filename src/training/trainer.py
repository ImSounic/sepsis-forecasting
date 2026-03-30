"""Training loop, utility score evaluation, and loss functions for sepsis prediction."""

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, TypeError, AttributeError):
    SummaryWriter = None


def compute_f1_score(labels, predictions):
    """Compute sample-level F1 score from binary labels and predictions."""
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_youden_j(labels, predictions):
    """Compute Youden's J index = sensitivity + specificity - 1.

    Ranges from -1 to 1. Higher is better; 0 means no discriminative ability.
    """
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def compute_utility_score(labels, predictions, patient_ids, hour_indices):
    """Compute PhysioNet 2019 utility score.

    Patient-level scoring based on timing of first positive prediction
    relative to sepsis onset. Normalized so 0 = all-negative baseline,
    1 = optimal detection.

    For sepsis patients (onset at t_sepsis):
      dt < -12:          too early, treated as false alarm (-0.05)
      -12 <= dt < -6:    early detection, linear credit [0, 1)
      -6 <= dt <= +3:    optimal window, full credit (1.0)
      dt > +3:           late detection, linearly decreasing to 0

    For non-sepsis patients:
      Each false positive hour incurs -0.05 penalty.

    Args:
        labels: Binary ground truth (0/1) per sample.
        predictions: Binary predictions (0/1) per sample.
        patient_ids: Patient ID per sample.
        hour_indices: Hour index within patient per sample.

    Returns:
        Normalized utility score (float).
    """
    # Group by patient: {pid: [(hour, label, pred), ...]}
    patient_data = defaultdict(list)
    for label, pred, pid, hour in zip(labels, predictions, patient_ids, hour_indices):
        patient_data[pid].append((int(hour), int(label), int(pred)))

    # Sort each patient by hour
    for pid in patient_data:
        patient_data[pid].sort()

    observed_utility = 0.0
    optimal_utility = 0.0
    baseline_utility = 0.0  # utility if we predict all-negative

    for pid, records in patient_data.items():
        hours_p = [r[0] for r in records]
        labels_p = [r[1] for r in records]
        preds_p = [r[2] for r in records]

        is_sepsis = 1 in labels_p

        if is_sepsis:
            t_sepsis = hours_p[labels_p.index(1)]
            optimal_utility += 1.0
            baseline_utility -= 2.0

            # Find first positive prediction
            first_pos = None
            for h, p in zip(hours_p, preds_p):
                if p == 1:
                    first_pos = h
                    break

            if first_pos is None:
                # Missed sepsis
                observed_utility -= 2.0
            else:
                dt = first_pos - t_sepsis
                if dt < -12:
                    observed_utility -= 0.05
                elif dt < -6:
                    observed_utility += (dt + 12) / 6.0
                elif dt <= 3:
                    observed_utility += 1.0
                else:
                    observed_utility += max(0.0, 1.0 - (dt - 3) / 9.0)
        else:
            # Non-sepsis: penalize false positives
            n_fp = sum(p for p in preds_p)
            observed_utility -= 0.05 * n_fp

    denominator = optimal_utility - baseline_utility
    if denominator == 0:
        return 0.0
    return (observed_utility - baseline_utility) / denominator


class SepsisTrainer:
    """Training loop with mixed precision, early stopping, and utility score evaluation."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Positive class weight for BCE
        pos_weight = config.get("pos_weight", 1.0)
        if pos_weight == "auto":
            pos_weight = self._compute_pos_weight()
        self.pos_weight = float(pos_weight)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=config.get("scheduler_patience", 5),
            factor=config.get("scheduler_factor", 0.5),
            min_lr=config.get("min_lr", 1e-5),
        )

        # Training params
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.patience = config.get("patience", 10)
        self.checkpoint_dir = config.get("checkpoint_dir", "outputs/models")
        self.log_dir = config.get("log_dir", "outputs/logs")

        # Mixed precision (only on CUDA)
        use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler() if use_amp else None

    def resume_from_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and scheduler state from a checkpoint.

        Returns tuple of (epoch, best_score, patience_counter).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resumed_epoch = checkpoint.get("epoch", 0)
        best_score = checkpoint.get("best_score", float("-inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(f"Resumed from checkpoint at epoch {resumed_epoch} "
              f"(utility={checkpoint.get('utility', 'N/A'):.4f}, "
              f"f1={checkpoint.get('f1', 'N/A'):.4f})")
        return resumed_epoch, best_score, patience_counter

    def _compute_pos_weight(self):
        """Compute pos_weight = num_negative / num_positive from training labels."""
        total, positive = 0, 0
        for _, labels, _, _, _ in self.train_loader:
            total += labels.numel()
            positive += labels.sum().item()
        if positive == 0:
            return 1.0
        return (total - positive) / positive

    def _compute_loss(self, logits, targets):
        """Weighted BCE with logits (numerically stable, autocast-safe)."""
        weight = torch.where(targets == 1, self.pos_weight, 1.0)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)

    def train_epoch(self):
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels, masks, _, _ in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)

            if n_batches == 0:
                print(f"  [DEBUG] Device: {self.device} | Features on: {features.device} | "
                      f"Batch shape: {features.shape} | AMP: {self.scaler is not None}")

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = self.model(features, padding_mask=masks)
                    loss = self._compute_loss(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(features, padding_mask=masks)
                loss = self._compute_loss(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self):
        """Evaluate on validation set.

        Returns:
            (val_loss, predictions, labels, patient_ids, hour_indices)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds, all_labels, all_pids, all_hours = [], [], [], []

        with torch.no_grad():
            for features, labels, masks, pids, hours in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)

                logits, _ = self.model(features, padding_mask=masks)
                loss = self._compute_loss(logits, labels)

                total_loss += loss.item()
                n_batches += 1

                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_pids.extend(list(pids))
                all_hours.extend(hours.tolist() if isinstance(hours, torch.Tensor) else hours)

        return total_loss / max(n_batches, 1), all_preds, all_labels, all_pids, all_hours

    def find_optimal_threshold(self, predictions, labels, patient_ids, hour_indices):
        """Search for threshold that maximizes utility score.

        Returns:
            (best_threshold, best_utility_score)
        """
        best_threshold = 0.5
        best_score = float("-inf")

        for threshold in np.arange(0.1, 0.91, 0.05):
            binary_preds = [1 if p >= threshold else 0 for p in predictions]
            score = compute_utility_score(labels, binary_preds, patient_ids, hour_indices)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return best_threshold, best_score

    def find_optimal_thresholds(self, predictions, labels, patient_ids, hour_indices):
        """Find optimal thresholds for utility, F1, and Youden's J.

        Uses 0.01 step for finer granularity (F1/Youden are more threshold-sensitive).

        Returns:
            dict with keys 'utility', 'f1', 'youden', each mapping to
            {'threshold': float, 'score': float}.
        """
        thresholds = np.arange(0.05, 0.91, 0.01)

        best = {
            "utility": {"threshold": 0.5, "score": float("-inf")},
            "f1": {"threshold": 0.5, "score": float("-inf")},
            "youden": {"threshold": 0.5, "score": float("-inf")},
        }

        for t in thresholds:
            binary_preds = [1 if p >= t else 0 for p in predictions]

            util = compute_utility_score(labels, binary_preds, patient_ids, hour_indices)
            if util > best["utility"]["score"]:
                best["utility"] = {"threshold": float(t), "score": util}

            f1 = compute_f1_score(labels, binary_preds)
            if f1 > best["f1"]["score"]:
                best["f1"] = {"threshold": float(t), "score": f1}

            youden = compute_youden_j(labels, binary_preds)
            if youden > best["youden"]["score"]:
                best["youden"] = {"threshold": float(t), "score": youden}

        return best

    def train(self, epochs, early_stop_metric="utility", checkpoint_subdir="gru",
              resume_epoch=0, resume_best_score=None, resume_patience=0):
        """Full training loop with early stopping and checkpointing.

        Args:
            epochs: Max number of training epochs.
            early_stop_metric: Metric for early stopping/checkpointing.
                One of 'utility', 'f1', 'youden'. Default 'utility'.
            checkpoint_subdir: Subdirectory under checkpoint_dir for saving
                the best model. Default 'gru'.
            resume_epoch: Epoch to resume from (0 = start fresh).
            resume_best_score: Best score from previous run (for resume).
            resume_patience: Patience counter from previous run (for resume).

        Returns:
            History dict with per-epoch metrics.
        """
        gru_checkpoint_dir = os.path.join(self.checkpoint_dir, checkpoint_subdir)
        os.makedirs(gru_checkpoint_dir, exist_ok=True)

        writer = None
        if SummaryWriter is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            writer = SummaryWriter(self.log_dir)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_utility": [],
            "val_f1": [],
            "val_youden": [],
            "threshold": [],
            "lr": [],
        }

        best_score = resume_best_score if resume_best_score is not None else float("-inf")
        patience_counter = resume_patience
        start_epoch = resume_epoch + 1

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch()

            val_loss, preds, labels, pids, hours = self.validate()
            all_thresholds = self.find_optimal_thresholds(preds, labels, pids, hours)

            # Extract scores for each metric
            utility = all_thresholds["utility"]["score"]
            f1 = all_thresholds["f1"]["score"]
            youden = all_thresholds["youden"]["score"]

            # Use selected metric for early stopping and scheduler
            current_score = all_thresholds[early_stop_metric]["score"]
            current_threshold = all_thresholds[early_stop_metric]["threshold"]

            self.scheduler.step(current_score)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_utility"].append(utility)
            history["val_f1"].append(f1)
            history["val_youden"].append(youden)
            history["threshold"].append(current_threshold)
            history["lr"].append(current_lr)

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Metrics/utility", utility, epoch)
                writer.add_scalar("Metrics/f1", f1, epoch)
                writer.add_scalar("Metrics/youden", youden, epoch)
                writer.add_scalar("Metrics/threshold", current_threshold, epoch)
                writer.add_scalar("LR", current_lr, epoch)

            metric_label = early_stop_metric.upper()
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Util: {utility:.4f} | F1: {f1:.4f} | Youden: {youden:.4f} | "
                f"Thr({metric_label}): {current_threshold:.2f} | LR: {current_lr:.1e}"
            )

            # Save last.pt every epoch (for resume on crash/disconnect)
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "utility": utility,
                "f1": f1,
                "youden": youden,
                "threshold": current_threshold,
                "early_stop_metric": early_stop_metric,
                "all_thresholds": all_thresholds,
                "best_score": best_score,
                "patience_counter": patience_counter,
            }
            torch.save(checkpoint_data, os.path.join(gru_checkpoint_dir, "last.pt"))

            # Early stopping + checkpointing based on selected metric
            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                torch.save(checkpoint_data, os.path.join(gru_checkpoint_dir, "best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if writer:
            writer.close()
        return history
