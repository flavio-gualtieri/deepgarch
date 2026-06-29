# src/deepgarch/data/natural_gas_loader.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping
import re

import numpy as np
import pandas as pd
import yfinance as yf


_DEFAULT_VAL_START = "2018-01-01"
_DEFAULT_TEST_START = "2021-01-01"
_DOWNLOAD_DIR = Path(__file__).parent / "downloaded"


@dataclass
class ExogenousSource:

    path: str | Path
    date_col: str = "date"
    prefix: str = ""
    release_lag_days: int = 1


class NaturalGasMarketData:

    def __init__(
        self,
        ticker: str = "NG=F",
        start: str = "2000-01-01",
        end: str | None = None,
        val_start: str = _DEFAULT_VAL_START,
        test_start: str = _DEFAULT_TEST_START,
        exogenous_sources: Iterable[ExogenousSource] | None = None,
        yahoo_aux_tickers: Mapping[str, str] | None = None,
    ) -> None:
        if pd.Timestamp(test_start) <= pd.Timestamp(val_start):
            raise ValueError(
                f"test_start ({test_start}) must be strictly after val_start ({val_start})."
            )

        self.ticker = ticker.upper()
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.val_start = pd.Timestamp(val_start)
        self.test_start = pd.Timestamp(test_start)
        self.exogenous_sources = list(exogenous_sources or [])
        self.yahoo_aux_tickers = dict(yahoo_aux_tickers or {})

        # Populated by load()
        self.prices: pd.Series | None = None
        self.returns: pd.Series | None = None
        self.frame: pd.DataFrame | None = None
        self.train: pd.DataFrame | None = None
        self.val: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> "NaturalGasMarketData":
        """Download/cache prices, build the modelling frame, and split."""
        ohlcv = self._download_target_ohlcv()
        frame = self._build_base_frame(ohlcv)

        for name, ticker in self.yahoo_aux_tickers.items():
            aux = self._download_aux_close(name=name, ticker=ticker)
            frame = frame.join(aux, how="left")

        for source in self.exogenous_sources:
            exog = self._read_exogenous_csv(source)
            frame = frame.join(exog, how="left")

        # Most external fundamentals are lower frequency. Forward-fill onto the
        # trading calendar; the feature pipeline will shift exogenous values to
        # avoid using information before it was observable.
        frame = frame.sort_index().ffill()

        self.frame = frame
        self.prices = frame["close"].rename(self.ticker)
        self.returns = frame["returns"].dropna().rename(f"{self.ticker}_returns")
        self.train, self.val, self.test = self._split_frame(frame)
        return self

    def summary(self) -> None:
        """Print a compact summary for each split."""
        self._require_loaded()
        assert self.train is not None and self.val is not None and self.test is not None

        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            returns = split["returns"].dropna()
            print(
                f"  {name:<6} {split.index[0].date()} → {split.index[-1].date()}"
                f"  ({len(split):>5} rows)"
                f"  ann.vol={self._ann_vol(returns):.2%}"
                f"  features={split.shape[1] - 1}"
            )

    # ------------------------------------------------------------------
    # Download/cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, suffix: str) -> Path:
        _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        start = pd.Timestamp(self.start).date().isoformat()
        end = pd.Timestamp(self.end).date().isoformat()
        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.upper())
        return _DOWNLOAD_DIR / f"{safe_ticker}_{suffix}_{start}_{end}.parquet"

    def _download_target_ohlcv(self) -> pd.DataFrame:
        cache_file = self._cache_path(self.ticker, "ohlcv")
        if cache_file.exists():
            cached = pd.read_parquet(cache_file).sort_index()
            if not cached.empty and cached.index[0] <= pd.Timestamp(self.start) + pd.Timedelta(days=5):
                return cached
            cache_file.unlink(missing_ok=True)

        print(f"[{self.ticker}] Downloading OHLCV {self.start} → {self.end} via yfinance…")
        raw = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no data for {self.ticker!r}.")

        raw = self._flatten_yfinance_columns(raw)
        cols = {c: c.lower().replace(" ", "_") for c in raw.columns}
        raw = raw.rename(columns=cols)

        required = {"open", "high", "low", "close"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"Downloaded data missing required columns: {sorted(missing)}")

        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in raw.columns]
        out = raw[keep].copy().sort_index()
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)

        earliest = out.index[0]
        if earliest > pd.Timestamp(self.start) + pd.Timedelta(days=5):
            raise ValueError(
                f"yfinance returned data starting {earliest.date()}, but start={self.start}."
            )

        out.to_parquet(cache_file)
        print(f"[{self.ticker}] Cached {len(out)} rows → {cache_file.name}")
        return out

    def _download_aux_close(self, name: str, ticker: str) -> pd.DataFrame:
        """Download an auxiliary Yahoo close series and return daily returns."""
        cache_file = self._cache_path(ticker, f"aux_{name}")
        if cache_file.exists():
            close = pd.read_parquet(cache_file)["close"].sort_index()
        else:
            print(f"[{ticker}] Downloading auxiliary series for {name}…")
            raw = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"yfinance returned no data for auxiliary ticker {ticker!r}.")
            raw = self._flatten_yfinance_columns(raw)
            close = raw["Close"].rename("close").sort_index()
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close.to_frame("close").to_parquet(cache_file)

        close = close.rename(f"{name}_close")
        ret = np.log(close / close.shift(1)).rename(f"{name}_ret")
        return pd.concat([close, ret], axis=1)

    @staticmethod
    def _flatten_yfinance_columns(raw: pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            # Single ticker downloads can still come back as a MultiIndex.
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
        return raw

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _build_base_frame(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        close = ohlcv["close"].astype(float)
        returns = np.log(close / close.shift(1))
        returns = returns - returns.mean(skipna=True)

        frame = ohlcv.copy()
        frame["returns"] = returns
        frame["log_price"] = np.log(close)

        # Range-based volatility proxy. Useful for daily futures data when you
        # do not have intraday realized volatility.
        frame["parkinson_var"] = (
            (np.log(frame["high"] / frame["low"]) ** 2) / (4.0 * np.log(2.0))
        )

        if "volume" in frame.columns:
            frame["log_volume"] = np.log1p(frame["volume"].replace(0, np.nan))
            frame["volume_chg"] = frame["log_volume"].diff()

        return frame

    @staticmethod
    def _read_exogenous_csv(source: ExogenousSource) -> pd.DataFrame:
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        if source.date_col not in df.columns:
            raise ValueError(f"{path} must contain date column {source.date_col!r}.")

        df[source.date_col] = pd.to_datetime(df[source.date_col])
        df = df.set_index(source.date_col).sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Keep numeric columns only; names get prefixed to avoid collisions.
        df = df.select_dtypes(include=[np.number]).copy()
        if source.prefix:
            df = df.add_prefix(source.prefix)

        # Conservative release lag to avoid look-ahead when the observation date
        # and publication date are not identical in the raw CSV.
        if source.release_lag_days:
            df.index = df.index + pd.Timedelta(days=source.release_lag_days)

        return df

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _split_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = frame[frame.index < self.val_start]
        val = frame[(frame.index >= self.val_start) & (frame.index < self.test_start)]
        test = frame[frame.index >= self.test_start]

        if train.empty:
            raise ValueError(f"Train split is empty. Move start earlier than {self.val_start.date()}.")
        if val.empty:
            raise ValueError("Validation split is empty. Adjust val_start/test_start.")
        if test.empty:
            raise ValueError("Test split is empty. Extend end or adjust test_start.")

        return train, val, test

    def _require_loaded(self) -> None:
        if self.frame is None:
            raise RuntimeError("Call .load() before accessing data splits.")

    @staticmethod
    def _ann_vol(returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(252))
