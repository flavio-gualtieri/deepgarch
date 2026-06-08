# src/deepgarch/train/trainer.py

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from ..models.garchnet import GARCHNet
from .config import TrainConfig


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    stopped_early: bool = False

    def plot_losses(self) -> None:
        """Plot training and validation loss curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — run: pip install matplotlib")
            return

        epochs = range(1, len(self.train_losses) + 1)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, self.train_losses, label="Train loss", linewidth=1.5)
        ax.plot(epochs, self.val_losses,   label="Val loss",   linewidth=1.5)
        ax.axvline(
            self.best_epoch + 1,
            color="grey", linestyle="--", linewidth=1,
            label=f"Best epoch ({self.best_epoch + 1})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative log-likelihood")
        ax.set_title("GARCHNet training")
        ax.legend()
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(self, model: GARCHNet, config: TrainConfig) -> None:
    
        self.model = model
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=self.config.patience // 2,
        )


    def _train_step(self, X: Tensor, returns: Tensor) -> float:

        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(X, returns)
        loss.backward()

        if self.config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip,
            )

        self.optimizer.step()
        return loss.item()


    def _eval_step(self, X: Tensor, returns: Tensor) -> float:

        self.model.eval()
        with torch.no_grad():
            loss = self.model(X, returns)
        return loss.item()


    def fit(
        self,
        X_train: Tensor,
        returns_train: Tensor,
        X_val: Tensor,
        returns_val: Tensor,
    ) -> TrainingResult:

        result = TrainingResult()
        checkpoint = Path(self.config.checkpoint_path)
        t0 = time.perf_counter()

        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):

            # --- training step ---
            train_loss = self._train_step(X_train, returns_train)
            result.train_losses.append(train_loss)

            # --- validation step (no gradients) ---
            val_loss = self._eval_step(X_val, returns_val)
            result.val_losses.append(val_loss)

            # --- scheduler ---
            self.scheduler.step(val_loss)

            # --- early stopping and checkpointing ---
            improvement = result.best_val_loss - val_loss
            if improvement > self.config.min_delta:
                result.best_val_loss = val_loss
                result.best_epoch    = epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), checkpoint)
            else:
                epochs_without_improvement += 1

            # --- logging ---
            if (epoch + 1) % self.config.log_every == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch + 1:>4d} | "
                    f"train {train_loss:>10.4f} | "
                    f"val {val_loss:>10.4f} | "
                    f"best {result.best_val_loss:>10.4f} | "
                    f"lr {lr:.2e} | "
                    f"patience {epochs_without_improvement}/{self.config.patience}"
                )

            if epochs_without_improvement >= self.config.patience:
                print(f"\n  Early stopping at epoch {epoch + 1}.")
                result.stopped_early = True
                break

        # Restore best weights
        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
            checkpoint.unlink()     # clean up — don't leave checkpoints on disk

        result.elapsed_seconds = time.perf_counter() - t0
        print(
            f"\n  Done. Best epoch: {result.best_epoch + 1} | "
            f"best val loss: {result.best_val_loss:.4f} | "
            f"elapsed: {result.elapsed_seconds:.1f}s"
        )
        return result