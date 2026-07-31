from __future__ import annotations

import numpy as np
import torch

from ..models.vol.garch import GARCH
from .baselines import ArchBaseline


class EWMA(ArchBaseline):
    """RiskMetrics-style EWMA variance:

        sigma2_t = decay * sigma2_{t-1} + (1 - decay) * r_{t-1}**2

    This is a GARCH(1,1) with omega=0, alpha=1-decay, beta=decay, so
    filtering reuses the existing torch GARCH recursion directly rather
    than a new GARCHFamily subclass. decay is fixed by convention (0.94
    is the RiskMetrics default for daily data), not estimated by MLE.
    """

    def __init__(self, decay: float = 0.94) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay!r}")

        self.decay = decay
        self._torch_garch: GARCH | None = None
        self._initial_variance: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "EWMA"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "EWMA(unfitted)"
        return f"EWMA(decay={self.decay:.4f}, persistence={self.persistence:.4f})"

    def fit(self, train_returns) -> "EWMA":
        train = torch.as_tensor(self._to_numpy(train_returns), dtype=torch.float32)

        self._initial_variance = train.var(unbiased=True)
        self._torch_garch = GARCH(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor([1.0 - self.decay], dtype=torch.float32),
            torch.tensor([self.decay], dtype=torch.float32),
            constraint="none",
        )
        self._fitted = True

        return self

    def filter(self, returns) -> np.ndarray:
        self._require_fitted()
        assert self._torch_garch is not None
        assert self._initial_variance is not None

        r = torch.as_tensor(self._to_numpy(returns), dtype=torch.float32)
        with torch.no_grad():
            filtered = self._torch_garch.filter(r, initial_variance=self._initial_variance)

        return filtered.detach().cpu().numpy()

    @property
    def persistence(self) -> float:
        self._require_fitted()
        assert self._torch_garch is not None
        return float(self._torch_garch.persistence)
