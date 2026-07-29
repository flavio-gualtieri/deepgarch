# src/deepgarch/models/cond_garchnet.py

import torch
import torch.nn as nn

from torch import Tensor


class ConditionalGARCHNet(nn.Module):

    _VALID_CONSTRAINTS = ("none", "positive", "stationary")

    def __init__(
        self,
        paramnet: nn.Module,
        p: int = 1,
        q: int = 1,
        constraint: str = "stationary",
        max_persistence: float = 0.995,
        s_max: float = 0.25,
        ablate_level_head: bool = False,
    ) -> None:
        super().__init__()

        if p != 1 or q != 1:
            raise NotImplementedError("Start with conditional GARCH(1,1).")
        if constraint not in self._VALID_CONSTRAINTS:
            raise ValueError(f"constraint must be one of {self._VALID_CONSTRAINTS}.")

        self.paramnet = paramnet
        self.p = p
        self.q = q
        self.constraint = constraint
        self.max_persistence = max_persistence
        self.s_max = s_max
        self.ablate_level_head = ablate_level_head
        self.register_buffer("_initial_variance", torch.tensor(float("nan")))
        self.register_buffer("_v0", torch.tensor(float("nan")))

        expected = 1 + q + p
        n_params = getattr(paramnet, "_n_params", None)
        if n_params != expected:
            raise ValueError(
                f"paramnet must output {expected} params for GARCH({p},{q}); "
                f"got {n_params}."
            )

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected = 1 + self.q + self.p
        if raw.ndim != 2 or raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw shape (T, {expected}); got {tuple(raw.shape)}."
            )

        v_raw = raw[:, 0]
        if self.ablate_level_head:
            v_raw = torch.zeros_like(v_raw)
        rho_raw = raw[:, 1 : 1 + self.q]
        phi_raw = raw[:, 1 + self.q : 1 + self.q + self.p]

        return v_raw, rho_raw, phi_raw

    def _constrain_path(
        self,
        v_raw: Tensor,
        rho_raw: Tensor,
        phi_raw: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if torch.isnan(self._v0):
            raise RuntimeError(
                "v0 is NaN"
            )
        
        rho = self.max_persistence * torch.sigmoid(rho_raw)
        alpha = rho * self.s_max * torch.sigmoid(phi_raw)
        beta = rho - alpha

        sigma_bar2 = torch.exp(self._v0 + v_raw)
        omega = (1 - rho[:, 0]) * sigma_bar2
        
        return omega, alpha, beta, rho

    def parameter_path(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raw = self.paramnet(X)
        v_raw, rho_raw, phi_raw = self._split(raw)
        return self._constrain_path(v_raw, rho_raw, phi_raw)

    def fit_initial_variance(self, returns_train: Tensor) -> None:
        """Seed the variance recursion from the train split only.

        The recursion needs a sigma2 value to start from at t=0. Recomputing
        that seed from whatever tensor happens to be passed to variance_path
        (e.g. a train+val+test concatenation used for warmed-up eval) leaks
        val/test variance into the initial condition. Fitting it once here,
        from train returns only, keeps the seed identical (and leak-free)
        across every later call — train, val, test, or the full history.
        """
        self._initial_variance = returns_train.var(unbiased=False).clamp_min(1e-8).detach()
        self._v0 = torch.log(self._initial_variance)

    def variance_path(
        self,
        returns: Tensor,
        omega: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tensor:
        if returns.ndim != 1:
            raise ValueError(f"returns must be 1D; got {tuple(returns.shape)}.")

        T = returns.shape[0]
        if T < 2:
            raise ValueError("Need at least two returns for a GARCH recursion.")

        if torch.isnan(self._initial_variance):
            raise RuntimeError(
                "Call fit_initial_variance(returns_train) before running the "
                "variance recursion."
            )

        sigma2_values = [self._initial_variance]

        for t in range(1, T):
            prev_sigma2 = sigma2_values[-1]
            sigma2_t = (
                omega[t - 1]
                + alpha[t - 1, 0] * returns[t - 1].pow(2)
                + beta[t - 1, 0] * prev_sigma2
            )
            sigma2_values.append(sigma2_t.clamp_min(1e-8))

        return torch.stack(sigma2_values)

    @staticmethod
    def negative_loglikelihood(returns: Tensor, sigma2: Tensor) -> Tensor:
        return 0.5 * torch.mean(torch.log(sigma2) + returns.pow(2) / sigma2)

    def diagnostics(self, X: Tensor, returns: Tensor) -> dict[str, Tensor]:
        omega, alpha, beta, rho = self.parameter_path(X)
        sigma2 = self.variance_path(returns, omega, alpha, beta)
        alpha_1 = alpha[:, 0]
        beta_1 = beta[:, 0]
        rho_1 = rho[:, 0]
        return {
            "omega": omega,
            "alpha": alpha_1,
            "beta": beta_1,
            "rho": rho_1,
            "sigma_bar2": omega / (1 - rho[:, 0]),
            "alpha_share": alpha_1 / rho_1,
            "persistence": alpha_1 + beta_1,
            "sigma2": sigma2,
            "sigma": torch.sqrt(sigma2),
        }

    def forecast_fixed_params(
        self,
        X: Tensor,
        returns: Tensor,
        h: int = 30,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if h <= 0:
            raise ValueError("h must be positive.")

        diag = self.diagnostics(X, returns)

        omega_T = diag["omega"][-1]
        alpha_T = diag["alpha"][-1]
        beta_T = diag["beta"][-1]
        rho_T = diag["rho"][-1]

        sigma2_next = (
            omega_T
            + alpha_T * returns[-1].pow(2)
            + beta_T * diag["sigma2"][-1]
        ).clamp_min(1e-8)

        forecasts = [sigma2_next]
        for _ in range(1, h):
            sigma2_next = (omega_T + rho_T * sigma2_next).clamp_min(1e-8)
            forecasts.append(sigma2_next)

        params_T = {
            "omega": omega_T,
            "alpha": alpha_T,
            "beta": beta_T,
            "rho": rho_T,
            "last_sigma2": diag["sigma2"][-1],
        }
        return torch.stack(forecasts), params_T

    def forward(self, X: Tensor, returns: Tensor) -> Tensor:
        if X.shape[0] != returns.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} timesteps but returns has {returns.shape[0]}."
            )
        omega, alpha, beta, _rho = self.parameter_path(X)
        sigma2 = self.variance_path(returns, omega, alpha, beta)
        return self.negative_loglikelihood(returns, sigma2)