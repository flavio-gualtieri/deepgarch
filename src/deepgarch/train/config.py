# src/deepgarch/train/trainer.py

"""
Training hyperparameters.

Kept in a separate module so experiments can be reproduced by saving /
logging the config, independent of the trainer logic.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    max_epochs:       int   = 200
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-4
    patience:         int   = 20
    min_delta:        float = 1e-4
    grad_clip:        float = 1.0
    checkpoint_path:  str   = "best_model.pt"
    log_every:        int   = 10