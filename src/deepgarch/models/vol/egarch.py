from __future__ import annotations

import math

import torch

from .garch_family import GARCHFamily

_E_ABS_Z = math.sqrt(2.0 / math.pi)  # E|z| for a standard normal innovation


class EGARCH(GARCHFamily):

    # Unlike GARCH/GJR-GARCH, omega/alpha/gamma live in log-variance space
    # and need no positivity constraint — only beta requires |beta| < 1 for
    # stationarity, so "positive" isn't a meaningful mode here.
    _VALID_CONSTRAINTS = ("none", "stationary")

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
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.constraint == "none":
            return self._omega_raw, self._alpha_raw, self._beta_raw, self._gamma_raw

        # "stationary": only beta needs bounding for |beta| < max_persistence.
        beta = torch.tanh(self._beta_raw) * self.max_persistence
        return self._omega_raw, self._alpha_raw, beta, self._gamma_raw

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
        # ARCH-type terms in EGARCH act on standardized residuals, so only
        # the log-variance AR coefficient beta governs shock persistence.
        return self.beta.abs().sum()

    def variance_equation(
        self, t: int, past_return: torch.Tensor, past_variance: torch.Tensor
    ) -> torch.Tensor:
        omega, alpha, beta, gamma = self._constrained_params()

        log_past_variance = torch.log(past_variance)
        z = past_return / torch.sqrt(past_variance)

        log_variance = (
            omega
            + (beta  * log_past_variance).sum()
            + (alpha * (z.abs() - _E_ABS_Z)).sum()
            + (gamma * z).sum()
        )

        return torch.exp(log_variance)
