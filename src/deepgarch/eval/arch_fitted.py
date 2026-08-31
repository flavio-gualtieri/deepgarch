from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Mapping

import torch

import numpy as np

from pandas import Series

from ..models.vol.garch import GARCHFamily
from .baselines import ArchBaseline


class ArchFittedBaseline(ArchBaseline):

    arch_model_config: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, scale: float = 100.0) -> None:
        super().__init__()
        self.scale = scale

        self._fit_result: Any | None = None
        self._initial_variance: torch.Tensor | None = None
        self._torch_garch: GARCHFamily | None = None

    @property
    def name(self) -> str:
        return type(self).__name__

    def fit(self, train_returns: Any) -> "ArchFittedBaseline":
        from arch import arch_model

        train = np.asarray(
            self._to_numpy(train_returns),
            dtype=np.float64,
        )

        # Common defaults can still be overridden by subclass configuration.
        model_config = {
            "mean": "Zero",
            "dist": "Normal",
            **self.arch_model_config,
        }

        model = arch_model(
            train * self.scale,
            **model_config,
        )
        self._fit_result = model.fit(disp="off")

        self._initial_variance = torch.as_tensor(
            train,
            dtype=torch.float32,
        ).var(unbiased=True)

        # Model-family-specific operation.
        self._torch_garch = self._garch_from_arch_params(self._fit_result.params)

        self._fitted = True

        return self

    def filter(self, returns: Any) -> np.ndarray:
        self._require_fitted()

        # These assertions help both static type checkers and future readers.
        assert self._torch_garch is not None
        assert self._initial_variance is not None

        returns_tensor = torch.as_tensor(
            self._to_numpy(returns),
            dtype=torch.float32,
        )

        with torch.no_grad():
            filtered = self._torch_garch.filter(
                returns_tensor,
                initial_variance=self._initial_variance,
            )

        return filtered.detach().cpu().numpy()

    @property
    def persistence(self) -> float:
        self._require_fitted()
        assert self._torch_garch is not None

        persistence = self._torch_garch.persistence

        if torch.is_tensor(persistence):
            return float(persistence.detach().cpu())

        return float(persistence)

    @abstractmethod
    def _garch_from_arch_params(
        self,
        params: Series,
    ) -> GARCHFamily:
        ...