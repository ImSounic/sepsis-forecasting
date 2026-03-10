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

    def train(self, epochs):
        """Full training loop with early stopping and checkpointing.

        Returns:
            History dict with per-epoch metrics.
        """
        gru_checkpoint_dir = os.path.join(self.checkpoint_dir, "gru")
        os.makedirs(gru_checkpoint_dir, exist_ok=True)

        writer = None
        if SummaryWriter is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            writer = SummaryWriter(self.log_dir)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_utility": [],
            "threshold": [],
            "lr": [],
        }

        best_utility = float("-inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()

            val_loss, preds, labels, pids, hours = self.validate()
            threshold, utility = self.find_optimal_threshold(preds, labels, pids, hours)

            self.scheduler.step(utility)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_utility"].append(utility)
            history["threshold"].append(threshold)
            history["lr"].append(current_lr)

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Metrics/utility", utility, epoch)
                writer.add_scalar("Metrics/threshold", threshold, epoch)
                writer.add_scalar("LR", current_lr, epoch)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Utility: {utility:.4f} | Thr: {threshold:.2f} | LR: {current_lr:.1e}"
            )

            # Early stopping + checkpointing
            if utility > best_utility:
                best_utility = utility
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "utility": utility,
                        "threshold": threshold,
                    },
                    os.path.join(gru_checkpoint_dir, "best.pt"),
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if writer:
            writer.close()
        return history
