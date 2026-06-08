# src/deepgarch/models/vol/garch_family.py

from abc import abstractmethod
from typing import List

import torch
from torch import Tensor

from .base import VolatilityModel


class GARCHFamily(VolatilityModel):

    @abstractmethod
    def variance_equation(
        self, t: int, past_return: Tensor, past_variance: Tensor
    ) -> Tensor:
        ...


    def initial_variance(self, returns: Tensor) -> Tensor:

        return returns.var(unbiased=True)


    def filter(self, returns: Tensor) -> Tensor:

        T = returns.shape[0]
        if T == 0:
            raise ValueError("filter() received an empty return series (T=0).")

        variance_list: list[Tensor] = []
        sigma2 = self.initial_variance(returns)

        for t in range(T):
            past_return   = returns[t - 1] if t > 0 else returns.new_zeros(())
            past_variance = sigma2 if t == 0 else variance_list[t - 1]
            sigma2        = self.variance_equation(t, past_return, past_variance)
            variance_list.append(sigma2)

        return torch.stack(variance_list)


    def loglikelihood(self, returns: Tensor) -> Tensor:

        variances = self.filter(returns)
        log_2pi = torch.log(
            torch.tensor(2 * torch.pi, dtype=returns.dtype, device=returns.device)
        )
        ll = -0.5 * (log_2pi + torch.log(variances) + returns ** 2 / variances)
        
        return ll.sum()


    def forecast(self, returns: Tensor, h: int) -> Tensor:

        variances = self.filter(returns)
        forecasts = torch.zeros(h, dtype=returns.dtype, device=returns.device)
        sigma2      = variances[-1]
        last_return = returns[-1]

        for i in range(h):
            sigma2        = self.variance_equation(len(returns) + i, last_return, sigma2)
            forecasts[i]  = sigma2
            last_return   = torch.sqrt(sigma2)   # E[|ε|] stand-in

        return forecasts