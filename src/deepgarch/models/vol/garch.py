# src/deepgarch/models/vol/garch.py

from .garch_family import GARCHFamily

import torch
import torch.nn.functional as F
from torch import Tensor


class GARCH(GARCHFamily):

    _VALID_CONSTRAINTS = ("none", "positive", "stationary")

    def __init__(
        self,
        omega: Tensor,
        alpha: Tensor,
        beta: Tensor,
        constraint: str = "stationary",
        max_persistence: float = 1.0,
    ) -> None:

        super().__init__()
        if constraint not in self._VALID_CONSTRAINTS:
            raise ValueError(
                f"constraint must be one of {self._VALID_CONSTRAINTS}, "
                f"got {constraint!r}"
            )
        self._omega_raw = omega
        self._alpha_raw = alpha
        self._beta_raw  = beta
        self.constraint = constraint
        self.max_persistence = max_persistence


    def _constrained_params(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.constraint == "none":
            return self._omega_raw, self._alpha_raw, self._beta_raw

        if self.constraint == "positive":
            return (
                F.softplus(self._omega_raw),
                F.softplus(self._alpha_raw),
                F.softplus(self._beta_raw),
            )

        # "stationary"
        # Append a zero-logit slack to the α and β logits, then softmax.
        # The slack class always receives positive probability, so the
        # α and β weights sum to strictly less than 1 (scaled by
        # max_persistence). No T dimension: one triple for the whole series.
        omega  = F.softplus(self._omega_raw)
        slack  = self._alpha_raw.new_zeros(1)
        logits = torch.cat([self._alpha_raw, self._beta_raw, slack], dim=-1)
        w      = torch.softmax(logits, dim=-1) * self.max_persistence

        q     = self._alpha_raw.shape[-1]
        alpha = w[:q]
        beta  = w[q:-1]
        return omega, alpha, beta


    @property
    def omega(self) -> Tensor:
        return self._constrained_params()[0]


    @property
    def alpha(self) -> Tensor:
        return self._constrained_params()[1]

    @property
    def beta(self) -> Tensor:
        return self._constrained_params()[2]


    def variance_equation(
        self, t: int, past_return: Tensor, past_variance: Tensor
    ) -> Tensor:

        omega, alpha, beta = self._constrained_params()
        arch_term  = (alpha * past_return ** 2).sum()
        garch_term = (beta  * past_variance).sum()

        return omega + arch_term + garch_term


    def stationarity_gap(self) -> Tensor:

        _, alpha, beta = self._constrained_params()
        return alpha.sum() + beta.sum()