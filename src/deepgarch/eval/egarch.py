from __future__ import annotations

import math

import torch

from pandas import Series

from ..models.vol.garch_family import GARCHFamily
from ..models.vol.egarch import EGARCH as EGARCHVariance
from .arch_fitted import ArchFittedBaseline


class EGARCH(ArchFittedBaseline):

    arch_model_config = {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}

    def __init__(self, scale: float = 100.0) -> None:
        super().__init__(scale=scale)

        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta:  float | None = None
        self.gamma: float | None = None

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "EGARCH(unfitted)"
        return (
            f"EGARCH(omega={self.omega:.2e}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, gamma={self.gamma:.4f}, "
            f"persistence={self.persistence:.4f})"
        )

    def _garch_from_arch_params(self, params: Series) -> GARCHFamily:
        omega_scaled = float(params["omega"])
        self.alpha = float(params["alpha[1]"])
        self.beta  = float(params["beta[1]"])
        self.gamma = float(params["gamma[1]"])

        # EGARCH's recursion is on log-variance: ln(S) = ln(s) + 2*ln(scale)
        # for scaled variance S = scale**2 * s, so undoing the return
        # scaling on omega is a beta-dependent shift, not a division like
        # GARCH's omega/scale**2.
        self.omega = omega_scaled + 2.0 * math.log(self.scale) * (self.beta - 1.0)

        return EGARCHVariance(
            torch.tensor(self.omega, dtype=torch.float32),
            torch.tensor([self.alpha], dtype=torch.float32),
            torch.tensor([self.beta], dtype=torch.float32),
            torch.tensor([self.gamma], dtype=torch.float32),
            constraint="none",
        )
