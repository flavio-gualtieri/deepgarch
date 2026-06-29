# src/garchnet/models/garchnet.py

import torch
import torch.nn as nn
from torch import Tensor

from .nn import ParamNet
from .vol.garch import GARCH


class GARCHNet(nn.Module):

    def __init__(self, paramnet: ParamNet, p: int = 1, q: int = 1,
                 constraint: str = "stationary", max_persistence: float = 1.0):
        super().__init__()
        if p != 1 or q != 1:
            raise NotImplementedError(
                "ConditionalGARCHNet currently supports only GARCH(1,1)."
            )
        self.paramnet = paramnet
        self.p = p
        self.q = q
        self.constraint = constraint
        self.max_persistence = max_persistence
 
        expected = 1 + q + p
        if paramnet._n_params != expected:
            raise ValueError(
                f"paramnet must output {expected} params for GARCH(p={p}, q={q}) "
                f"(1 omega + {q} alpha + {p} beta), but outputs {paramnet._n_params}."
            )
        
    def _negative_loglikelihood(self, returns: Tensor, sigma2: Tensor) -> Tensor:
        return 0.5 * torch.mean(
            torch.log(sigma2) + returns.pow(2) / sigma2
        )
        
    def _constrain_path(self,
                        omega_raw: Tensor,   # (T,)
                        alpha_raw: Tensor,   # (T, 1)
                        beta_raw: Tensor,    # (T, 1)
                        ) -> tuple[Tensor, Tensor, Tensor]:
        eps = 1e-8

        omega = torch.nn.functional.softplus(omega_raw) + eps

        if self.constraint == "none":
            return omega_raw, alpha_raw, beta_raw

        if self.constraint == "positive":
            alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)
            return omega, alpha, beta

        if self.constraint == "stationary":
            # alpha_raw and beta_raw are both (T, 1) for GARCH(1,1)
            slack = alpha_raw.new_zeros(alpha_raw.shape[0], 1)

            logits = torch.cat([alpha_raw, beta_raw, slack], dim=-1)
            weights = torch.softmax(logits, dim=-1) * self.max_persistence

            alpha = weights[:, : self.q]
            beta = weights[:, self.q : self.q + self.p]

            return omega, alpha, beta

        raise ValueError(f"Unknown constraint: {self.constraint}")

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if raw.ndim != 2:
            raise ValueError(
                f"Expected raw params with shape (T, {1 + self.q + self.p}), "
                f"got {tuple(raw.shape)}."
            )

        expected = 1 + self.q + self.p
        if raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw.shape[1] == {expected}, got {raw.shape[1]}."
            )

        omega = raw[:, 0]
        alpha = raw[:, 1 : 1 + self.q]
        beta = raw[:, 1 + self.q : 1 + self.q + self.p]

        return omega, alpha, beta
    
    def _variance_path(self,
                       returns: Tensor,   # (T,)
                       omega: Tensor,     # (T,)
                       alpha: Tensor,     # (T, q)
                       beta: Tensor,      # (T, p)
                       ) -> Tensor:
        T = returns.shape[0]

        sigma2 = torch.empty_like(returns)
        sigma2[0] = returns.var(unbiased=False).clamp_min(1e-8)

        for t in range(1, T):
            arch = alpha[t - 1, 0] * returns[t - 1].pow(2)
            garch = beta[t - 1, 0] * sigma2[t - 1]

            sigma2[t] = omega[t - 1] + arch + garch

        return sigma2.clamp_min(1e-8)

    def parameter_path(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.paramnet(embeddings)
        omega_raw, alpha_raw, beta_raw = self._split(raw)
        omega, alpha, beta = self._constrain_path(omega_raw, alpha_raw, beta_raw)
        return omega, alpha, beta
    
    def forward(self, embeddings: Tensor, returns: Tensor) -> Tensor:
        if embeddings.shape[0] != returns.shape[0]:
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} timesteps but "
                f"returns has {returns.shape[0]}; they must match."
            )

        omega, alpha, beta = self.parameter_path(embeddings)

        sigma2 = self._variance_path(returns=returns, omega=omega,
                                     alpha=alpha, beta=beta)

        return self._negative_loglikelihood(returns, sigma2)
    
    def diagnostics(self, X, returns) -> dict[str, Tensor]:
        omega, alpha, beta = self.parameter_path(X)
        sigma2 = self._variance_path(returns, omega, alpha, beta)

        return {
            "omega": omega,
            "alpha": alpha[:, 0],
            "beta": beta[:, 0],
            "persistence": alpha[:, 0] + beta[:, 0],
            "sigma2": sigma2,
            "sigma": torch.sqrt(sigma2),
        }