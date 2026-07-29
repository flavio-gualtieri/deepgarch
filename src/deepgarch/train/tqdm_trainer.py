# src/deepgarch/train/tqdm_trainer.py

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from .trainer import Trainer, TrainingResult

_BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"


class TqdmTrainer(Trainer):
    """Trainer that reports progress via a tqdm bar instead of periodic prints."""

    def _on_start(self) -> None:
        self._pbar = tqdm(
            total=self.config.max_epochs,
            desc="training",
            unit="ep",
            bar_format=_BAR_FMT,
            dynamic_ncols=True,
        )

    def _on_epoch(self, epoch, train_loss, val_loss, lr, best_val_loss, epochs_without_improvement) -> None:
        self._pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            lr=f"{lr:.2e}",
            best=f"{best_val_loss:.4f}",
            refresh=False,
        )
        self._pbar.update(1)

    def _on_early_stop(self, epoch, best_val_loss) -> None:
        self._pbar.write(f"  early stop at epoch {epoch + 1} (best val={best_val_loss:.4f})")

    def _on_end(self) -> None:
        self._pbar.close()

    def _on_finish(self, result: TrainingResult) -> None:
        pass
