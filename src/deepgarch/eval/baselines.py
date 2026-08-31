from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

import numpy as np

from pandas import Series


class ArchBaseline(ABC):

    def __init__(self) -> None:
        self._fitted: bool = False

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"{self.name}(unfitted)"
        return (
            f"{self.name}(persistence={self.persistence:.4f})"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def persistence(self) -> float:
        ...

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, train_returns: Any) -> "ArchBaseline":
        ...

    @abstractmethod
    def filter(self, returns: Any) -> np.ndarray:
        ...

    @staticmethod
    def _to_numpy(x: torch.Tensor | Series[Any]) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before filter().")