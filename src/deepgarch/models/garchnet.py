# src/garchnet/models/garchnet.py

import torch.nn as nn
from torch import Tensor

from .nn import ParamNet
from .vol.garch import GARCH

class GARCHNet(nn.Module):

    def __init__(
            self,
            paramnet: ParamNet,
            p: int = 1,
            q: int = 1,
            constraint: str = "stationary",
            max_persistence: float = 1.0
    ):

        super().__init__()
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
        
    
    def _pool(self, embeddings: Tensor) -> Tensor:
        """Pool T x embedding_dim -> embedding_dim by averaging over time."""
        return embeddings.mean(dim=0)
    

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    
        omega = raw[0]                          # ()
        alpha = raw[1 : 1 + self.q]            # (q,)
        beta  = raw[1 + self.q : 1 + self.q + self.p]  # (p,)
        return omega, alpha, beta
    

    def build_garch(self, embeddings: Tensor) -> GARCH:

        context = self._pool(embeddings)
        raw = self.paramnet(context)
        omega, alpha, beta = self._split(raw)

        return GARCH(
            omega,
            alpha,
            beta,
            constraint=self.constraint,
            max_persistence=self.max_persistence,
        )
    
    
    def forward(self, embeddings: Tensor, returns: Tensor) -> Tensor:
    
        if embeddings.shape[0] != returns.shape[0]:
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} timesteps but "
                f"returns has {returns.shape[0]}; they must match."
            )
        garch = self.build_garch(embeddings)

        return -garch.loglikelihood(returns)