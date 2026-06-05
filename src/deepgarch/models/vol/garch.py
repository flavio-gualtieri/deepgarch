# src/deepgarch/models/vol/garch.py

from .garch_family import GARCHFamily

import torch
import torch.nn.functional as F
from torch import Tensor

class GARCH(GARCHFamily):
    
    def __init__(self, omega: float, alpha: float, beta: float):
    
        super().__init__()
        self._omega_raw = omega
        self._alpha_raw = alpha
        self._beta_raw = beta


    @property
    def omega(self) -> Tensor:
        return F.softplus(self._omega_raw)
    

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self._alpha_raw)
    

    @property
    def beta(self) -> Tensor:
        return F.softplus(self._beta_raw)
    

    def _params_at(self, t: int) -> tuple[Tensor, Tensor, Tensor]:
    
        omega = self.omega[t] if self.omega.dim() == 1 else self.omega
        alpha = self.alpha[t] if self.alpha.dim() == 2 else self.alpha
        beta  = self.beta[t]  if self.beta.dim()  == 2 else self.beta
    
        return omega, alpha, beta
    
    def variance_equation(self, t: int, past_return: Tensor, past_variance: Tensor) -> Tensor:

        omega, alpha, beta = self._params_at(t)
 
        arch_term = (alpha * past_return ** 2).sum()
        garch_term = (beta * past_variance).sum()
 
        return omega + arch_term + garch_term
 
    def stationarity_gap(self) -> Tensor:
        return self.alpha.sum(-1) + self.beta.sum(-1)