# src/deepgarch/features/natural_gas_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureSpec:
    """A computed feature column."""

    name: str
    values: pd.Series


class NaturalGasFeaturePipeline:

    _BASE_COLUMNS = {
        "open", "high", "low", "close", "adj_close", "returns", "log_price",
        "parkinson_var", "volume", "log_volume", "volume_chg",
    }

    def __init__(
        self,
        return_windows: Sequence[int] = (5, 10, 21, 63),
        exogenous_columns: Iterable[str] | None = None,
        exogenous_lag: int = 1,
        include_seasonality: bool = True,
    ) -> None:
        if not return_windows:
            raise ValueError("return_windows must contain at least one window.")
        if exogenous_lag < 0:
            raise ValueError("exogenous_lag must be non-negative.")

        self.return_windows = tuple(int(w) for w in return_windows)
        self.exogenous_columns = list(exogenous_columns) if exogenous_columns is not None else None
        self.exogenous_lag = int(exogenous_lag)
        self.include_seasonality = include_seasonality

        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("Call fit() before reading feature_names.")
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def fit(self, frame: pd.DataFrame) -> "NaturalGasFeaturePipeline":
        raw = self._compute_raw(frame)
        self._means = raw.mean()
        self._stds = raw.std().replace(0, 1)
        self._feature_names = raw.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> Tensor:
        self._require_fitted()
        raw = self._compute_raw(frame)

        # Preserve the training-time feature order and fill newly absent columns
        # with zero after normalization.
        raw = raw.reindex(columns=self._feature_names)
        normalized = (raw - self._means) / self._stds
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return torch.tensor(normalized.values, dtype=torch.float32)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return normalized features as a dataframe for debugging/inspection."""
        self._require_fitted()
        raw = self._compute_raw(frame).reindex(columns=self._feature_names)
        normalized = (raw - self._means) / self._stds
        return normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def fit_transform(self, frame: pd.DataFrame) -> Tensor:
        return self.fit(frame).transform(frame)

    # ------------------------------------------------------------------
    # Raw feature computation
    # ------------------------------------------------------------------

    def _compute_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "returns" not in frame.columns:
            raise ValueError("frame must contain a 'returns' column.")

        ret = frame["returns"].astype(float)
        specs: list[FeatureSpec] = []

        # GARCH-friendly return state.
        specs.append(FeatureSpec("ret_lag1", ret.shift(1)))
        specs.append(FeatureSpec("eps2_lag1", (ret ** 2).shift(1)))
        specs.append(FeatureSpec("abs_ret_lag1", ret.abs().shift(1)))

        for w in self.return_windows:
            specs.extend([
                FeatureSpec(f"rvol_{w}d", ret.rolling(w).std().shift(1)),
                FeatureSpec(f"abs_ret_{w}d", ret.abs().rolling(w).mean().shift(1)),
                FeatureSpec(f"mom_{w}d", ret.rolling(w).sum().shift(1)),
            ])

        # Daily range-based volatility proxy from OHLC data.
        if "parkinson_var" in frame.columns:
            specs.append(FeatureSpec("parkinson_var_lag1", frame["parkinson_var"].shift(1)))
            for w in self.return_windows:
                specs.append(
                    FeatureSpec(
                        f"parkinson_var_{w}d",
                        frame["parkinson_var"].rolling(w).mean().shift(1),
                    )
                )

        # Liquidity/participation features.
        if "log_volume" in frame.columns:
            specs.append(FeatureSpec("log_volume_lag1", frame["log_volume"].shift(1)))
        if "volume_chg" in frame.columns:
            specs.append(FeatureSpec("volume_chg_lag1", frame["volume_chg"].shift(1)))

        # Natural-gas seasonal structure: winter heating and summer cooling regimes.
        if self.include_seasonality:
            day = frame.index.dayofyear.astype(float)
            specs.append(FeatureSpec("season_sin", pd.Series(np.sin(2 * np.pi * day / 365.25), index=frame.index)))
            specs.append(FeatureSpec("season_cos", pd.Series(np.cos(2 * np.pi * day / 365.25), index=frame.index)))

        # External features: storage, weather, CFTC positioning, production,
        # LNG feedgas, curve spreads, cross-market returns, etc.
        for col in self._resolve_exogenous_columns(frame):
            s = frame[col].astype(float)
            specs.append(FeatureSpec(f"{col}_lag{self.exogenous_lag}", s.shift(self.exogenous_lag)))
            specs.append(FeatureSpec(f"{col}_chg_lag{self.exogenous_lag}", s.diff().shift(self.exogenous_lag)))

            # A 63-trading-day percentile-like z-score captures tight/loose regimes
            # without imposing a long full-sample lookahead statistic.
            roll_mean = s.rolling(63, min_periods=20).mean()
            roll_std = s.rolling(63, min_periods=20).std().replace(0, np.nan)
            specs.append(
                FeatureSpec(
                    f"{col}_z63_lag{self.exogenous_lag}",
                    ((s - roll_mean) / roll_std).shift(self.exogenous_lag),
                )
            )

        raw = pd.DataFrame({spec.name: spec.values for spec in specs}, index=frame.index)
        return raw

    def _resolve_exogenous_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.exogenous_columns is not None:
            missing = [c for c in self.exogenous_columns if c not in frame.columns]
            if missing:
                raise ValueError(f"Missing exogenous columns: {missing}")
            return self.exogenous_columns

        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in self._BASE_COLUMNS]

    def _require_fitted(self) -> None:
        if self._means is None or self._stds is None or self._feature_names is None:
            raise RuntimeError("Call fit() or fit_transform() before transform().")


def default_natural_gas_feature_pipeline() -> NaturalGasFeaturePipeline:
    """Convenience factory for a first experiment."""
    return NaturalGasFeaturePipeline(
        return_windows=(5, 10, 21, 63),
        exogenous_lag=1,
        include_seasonality=True,
    )
