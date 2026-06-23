# src/deepgarch/features/pipeline.py

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from torch import Tensor



class Feature(ABC):
    @abstractmethod
    def compute(self, returns: pd.Series) -> pd.Series:
        ...


    @property
    @abstractmethod
    def name(self) -> str:
        """A short human-readable name, used as the column label."""
        ...



class RealizedVolatility(Feature):
    def __init__(self, window: int) -> None:
        self.window = window


    @property
    def name(self) -> str:
        return f"rvol_{self.window}d"


    def compute(self, returns: pd.Series) -> pd.Series:
        return returns.rolling(self.window).std().shift(1)



class LaggedSquaredReturn(Feature):
    def __init__(self, lag: int = 1) -> None:
        self.lag = lag


    @property
    def name(self) -> str:
        return f"eps2_lag{self.lag}"


    def compute(self, returns: pd.Series) -> pd.Series:
        return (returns ** 2).shift(self.lag)



class ReturnMomentum(Feature):
    def __init__(self, window: int) -> None:
        self.window = window


    @property
    def name(self) -> str:
        return f"mom_{self.window}d"


    def compute(self, returns: pd.Series) -> pd.Series:
        return returns.rolling(self.window).sum().shift(1)



class AbsReturnMean(Feature):
    def __init__(self, window: int) -> None:
        self.window = window


    @property
    def name(self) -> str:
        return f"abs_ret_{self.window}d"


    def compute(self, returns: pd.Series) -> pd.Series:
        return returns.abs().rolling(self.window).mean().shift(1)



class FeaturePipeline:
    def __init__(self, features: list[Feature], min_norm_window: int = 63) -> None:
        if not features:
            raise ValueError("FeaturePipeline requires at least one Feature.")
        self.features = features
        self.min_norm_window = min_norm_window
        self._means: pd.Series | None = None   # (n_features,)
        self._stds:  pd.Series | None = None   # (n_features,)


    @property
    def n_features(self) -> int:
        return len(self.features)


    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


    def fit(self, returns: pd.Series) -> "FeaturePipeline":
        raw = self._compute_raw(returns)
        self._means = raw.mean()
        self._stds  = raw.std().replace(0, 1)   # avoid divide-by-zero for constant features
        
        return self


    def transform(self, returns: pd.Series) -> Tensor:
        self._require_fitted()
        raw = self._compute_raw(returns)
        normalized = (raw - self._means) / self._stds
        normalized = normalized.ffill().fillna(0.0)

        return torch.tensor(normalized.values, dtype=torch.float32)


    def fit_transform(self, returns: pd.Series) -> Tensor:
        return self.fit(returns).transform(returns)


    def _compute_raw(self, returns: pd.Series) -> pd.DataFrame:
        columns = {f.name: f.compute(returns) for f in self.features}
        return pd.DataFrame(columns, index=returns.index)


    def _require_fitted(self) -> None:
        if self._means is None:
            raise RuntimeError("Call fit() or fit_transform() before transform().")