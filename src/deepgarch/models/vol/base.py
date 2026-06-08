# src/deepgarch/models/vol/base.py

from abc import ABC, abstractmethod

from torch import Tensor


class VolatilityModel(ABC):

    @abstractmethod
    def filter(self, returns: Tensor) -> Tensor:
        ...


    @abstractmethod
    def loglikelihood(self, returns: Tensor) -> Tensor:
        ...


    @abstractmethod
    def forecast(self, returns: Tensor, h: int) -> Tensor:
        ...