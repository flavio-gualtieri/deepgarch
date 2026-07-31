from __future__ import annotations

import torch

from pandas import Series

from ..models.vol.garch import GARCHFamily, GARCH
from .arch_fitted import ArchFittedBaseline


class StaticGARCH(ArchFittedBaseline):

    arch_model_config = {"vol": "Garch", "p": 1, "q": 1}

    def __init__(self, scale: float = 100.0) -> None:
        super().__init__(scale=scale)

        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta:  float | None = None

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "StaticGARCH(unfitted)"
        return (
            f"StaticGARCH(omega={self.omega:.2e}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f})"
        )
    
    def _garch_from_arch_params(self, params: Series) -> GARCHFamily:
        self.omega = float(params["omega"]) / self.scale ** 2
        self.alpha = float(params["alpha[1]"])
        self.beta  = float(params["beta[1]"])

        garch = GARCH(
            torch.tensor(self.omega, dtype=torch.float32),
            torch.tensor([self.alpha], dtype=torch.float32),
            torch.tensor([self.beta], dtype=torch.float32),
            constraint="none",
        )

        return garch