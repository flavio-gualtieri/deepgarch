# src/deepgarch/config.py

"""
YAML-driven configuration for the unified run.py entry point.

Each section maps 1:1 onto the constructor kwargs of the component it
configures (MarketData, FeaturePipeline, ParamNet/ConditionalGARCHNet,
TrainConfig), so a run is fully described by one YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .train.config import TrainConfig


@dataclass
class DataConfig:
    ticker: str
    start: str
    val_start: str
    test_start: str
    end: str | None = None
    yahoo_aux_tickers: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    return_windows: list[int] = field(default_factory=lambda: [5, 10, 21, 63])
    exogenous_lag: int = 1
    include_seasonality: bool = True
    include_eia_calendar: bool = True


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.10
    p: int = 1
    q: int = 1
    rho_init: float = 4.95
    phi_init: float = -1.15
    constraint: str = "stationary"
    max_persistence: float = 0.995
    s_max: float = 0.25
    v_max: float = 3.0
    ablate_level_head: bool = False


@dataclass
class ForecastConfig:
    horizon: int = 30


@dataclass
class OutputConfig:
    market: str
    dir: str
    events: dict[str, str] = field(default_factory=dict)
    regime_split: str | None = None


# Which split run.py scores its metrics on. The variance recursion always runs
# over the full train+val+test path; this only selects the window that is
# summarised. "test" reproduces the historical behaviour.
EVAL_SPLITS = ("train", "val", "test")


@dataclass
class RunConfig:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    forecast: ForecastConfig
    output: OutputConfig
    seed: int = 42
    eval_split: str = "test"

    def __post_init__(self) -> None:
        if self.eval_split not in EVAL_SPLITS:
            raise ValueError(
                f"eval_split must be one of {EVAL_SPLITS}, got {self.eval_split!r}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            data=DataConfig(**raw["data"]),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            forecast=ForecastConfig(**raw.get("forecast", {})),
            output=OutputConfig(**raw["output"]),
            seed=raw.get("seed", 42),
            eval_split=raw.get("eval_split", "test"),
        )
