# src/deepgarch/data/loader.py
import numpy as np
import pandas as pd
import yfinance as yf

_DEFAULT_VAL_START  = "2018-01-01"
_DEFAULT_TEST_START = "2021-01-01"


class MarketData:
    def __init__(
        self,
        ticker: str,
        start: str,
        end: str | None = None,
        val_start: str = _DEFAULT_VAL_START,
        test_start: str = _DEFAULT_TEST_START,
    ) -> None:
        if pd.Timestamp(test_start) <= pd.Timestamp(val_start):
            raise ValueError(
                f"test_start ({test_start}) must be after val_start ({val_start})."
            )
        self.ticker = ticker
        self.start = start
        self.end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.val_start = pd.Timestamp(val_start)
        self.test_start = pd.Timestamp(test_start)
        # Populated by load()
        self.prices: pd.Series | None = None
        self.returns: pd.Series | None = None
        self.train: pd.Series | None = None
        self.val: pd.Series | None = None
        self.test: pd.Series | None = None

    def load(self) -> "MarketData":
        self.prices  = self._download()
        self.returns = self._compute_returns(self.prices)
        self.train, self.val, self.test = self._split(self.returns)
        return self

    def summary(self) -> None:
        self._require_loaded()
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            print(
                f"  {name:<6} {split.index[0].date()} → {split.index[-1].date()}"
                f"  ({len(split)} obs)"
                f"  ann.vol={self._ann_vol(split):.1%}"
            )

    def _download(self) -> pd.Series:
        try:
            raw = yf.Ticker(self.ticker).history(
                start=self.start,
                end=self.end,
                auto_adjust=True,
            )
        except Exception as exc:
            if "rate" in str(exc).lower() or "too many" in str(exc).lower():
                raise RuntimeError(
                    "Yahoo Finance rate limit hit. Wait ~60 s and try again."
                ) from exc
            raise
        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {self.ticker!r} "
                f"between {self.start} and {self.end}."
            )
        prices = raw["Close"].copy()
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert(None)
        prices.name = self.ticker
        prices.sort_index(inplace=True)
        return prices

    def _compute_returns(self, prices: pd.Series) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        demeaned = log_returns - log_returns.mean()
        demeaned.name = f"{self.ticker}_returns"
        return demeaned

    def _split(
        self, returns: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        train = returns[returns.index < self.val_start]
        val   = returns[(returns.index >= self.val_start) & (returns.index < self.test_start)]
        test  = returns[returns.index >= self.test_start]
        if train.empty:
            raise ValueError(
                f"Train split is empty. Check that start ({self.start}) is "
                f"well before val_start ({self.val_start.date()})."
            )
        if val.empty or test.empty:
            raise ValueError(
                "Val or test split is empty. Extend the date range or "
                "adjust val_start / test_start."
            )
        return train, val, test

    def _require_loaded(self) -> None:
        if self.returns is None:
            raise RuntimeError("Call .load() before accessing data.")

    @staticmethod
    def _ann_vol(returns: pd.Series) -> float:
        return float(returns.std() * np.sqrt(252))