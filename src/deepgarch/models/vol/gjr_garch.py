from __future__ import annotations

import torch

import torch.nn.functional as F

from .garch_family import GARCHFamily


class GJRGARCH(GARCHFamily):

    _VALID_CONSTRAINTS = ("none", "positive", "stationary")

    def __init__(
        self,
        omega: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        constraint: str = "stationary",
        max_persistence: float = 1.0,   
    ) -> None:
        super().__init__()
        if constraint not in self._VALID_CONSTRAINTS:
            raise ValueError(
                f"constraint must be one of {self._VALID_CONSTRAINTS}, "
                f"got {constraint!r}"
            )

        if not 0.0 < max_persistence <= 1.0:
            raise ValueError("max_persistence must be in (0, 1].")
        
        self._omega_raw = omega
        self._alpha_raw = alpha
        self._beta_raw  = beta
        self._gamma_raw = gamma

        self.constraint = constraint
        self.max_persistence = max_persistence

    def _constrained_params(
            self
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        if self.constraint == "none":
            return (
                self._omega_raw,
                self._alpha_raw,
                self._beta_raw,
                self._gamma_raw,
            )

        if self.constraint == "positive":
            return (
                F.softplus(self._omega_raw),
                F.softplus(self._alpha_raw),
                F.softplus(self._beta_raw),
                F.softplus(self._gamma_raw),
            )

        omega = F.softplus(self._omega_raw)

        slack = self._alpha_raw.new_zeros(1)

        logits = torch.cat(
            [
                self._alpha_raw,
                self._beta_raw,
                self._gamma_raw,
                slack,
            ],
            dim=-1,
        )

        weights = torch.softmax(logits, dim=-1)
        weights = weights * self.max_persistence

        q = self._alpha_raw.numel()
        p = self._beta_raw.numel()

        alpha = weights[:q]
        beta = weights[q : q + p]

        # gamma / 2 receives this part of the persistence budget.
        gamma = 2.0 * weights[q + p : -1]

        return omega, alpha, beta, gamma

    @property
    def omega(self) -> torch.Tensor:
        return self._constrained_params()[0]

    @property
    def alpha(self) -> torch.Tensor:
        return self._constrained_params()[1]

    @property
    def beta(self) -> torch.Tensor:
        return self._constrained_params()[2]

    @property
    def gamma(self) -> torch.Tensor:
        return self._constrained_params()[3]

    @property
    def persistence(self) -> torch.Tensor:
        return (
            self.alpha.sum()
            + self.beta.sum()
            + 0.5 * self.gamma.sum()
        )

    def variance_equation(
        self, t: int, past_return: torch.Tensor, past_variance: torch.Tensor
    ) -> torch.Tensor:
        omega, alpha, beta, gamma = self._constrained_params()

        i_t = (past_return < 0).to(dtype=past_return.dtype)

        arch_term = (alpha * past_return.square()).sum()
        garch_term = (beta  * past_variance).sum()
        asym_term = (
            gamma * i_t * past_return.square()
        ).sum()

        return omega + arch_term + garch_term + asym_term