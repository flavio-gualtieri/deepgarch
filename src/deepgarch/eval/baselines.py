import numpy as np
import torch
from torch import Tensor

from ..models.vol.garch import GARCH


def _to_numpy(x) -> np.ndarray:
    """Coerce a tensor / Series / array-like to a 1-D numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class StaticGARCH:

    def __init__(self, scale: float = 100.0) -> None:
        self.scale = scale
        self.omega: float | None = None
        self.alpha: float | None = None
        self.beta:  float | None = None
        self._fit_result = None

    def fit(self, train_returns) -> "StaticGARCH":

        from arch import arch_model

        y = _to_numpy(train_returns) * self.scale
        model = arch_model(y, mean="Zero", vol="Garch", p=1, q=1, dist="Normal")
        res = model.fit(disp="off")

        p = res.params
        self.omega = float(p["omega"]) / self.scale ** 2
        self.alpha = float(p["alpha[1]"])
        self.beta  = float(p["beta[1]"])
        self._fit_result = res
        return self

    def filter(self, returns) -> np.ndarray:

        self._require_fitted()
        r = torch.tensor(_to_numpy(returns), dtype=torch.float32)
        garch = GARCH(
            torch.tensor(self.omega, dtype=torch.float32),
            torch.tensor([self.alpha], dtype=torch.float32),
            torch.tensor([self.beta], dtype=torch.float32),
            constraint="none",
        )
        with torch.no_grad():
            return garch.filter(r).numpy()

    @property
    def persistence(self) -> float:
        """α + β — the volatility persistence of the fitted model."""
        self._require_fitted()
        return self.alpha + self.beta

    def __repr__(self) -> str:
        if self.omega is None:
            return "StaticGARCH(unfitted)"
        return (
            f"StaticGARCH(omega={self.omega:.2e}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f})"
        )

    def _require_fitted(self) -> None:
        if self.omega is None:
            raise RuntimeError("Call fit() before filter().")