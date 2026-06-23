"""
src/deepgarch/data/loader.py
Market data loader backed by yfinance with local parquet caching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_VAL_START  = "2018-01-01"
_DEFAULT_TEST_START = "2021-01-01"
_DOWNLOAD_DIR = Path(__file__).parent / "downloaded"


class MarketData:
    """Download, cache, and split daily equity returns for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"SPY"``, ``"^GSPC"``).
    start : str
        Earliest date to request, ISO format ``"YYYY-MM-DD"``.
    end : str | None
        Latest date to request (inclusive).  Defaults to today.
    val_start : str
        First date of the validation split.
    test_start : str
        First date of the test split.  Must be strictly after *val_start*.

    Examples
    --------
    >>> md = MarketData("SPY", start="2000-01-01").load()
    >>> md.summary()
    """

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
                f"test_start ({test_start}) must be strictly after val_start ({val_start})."
            )

        self.ticker     = ticker.upper()
        self.start      = start
        self.end        = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.val_start  = pd.Timestamp(val_start)
        self.test_start = pd.Timestamp(test_start)

        # Populated by load()
        self.prices:  pd.Series | None = None
        self.returns: pd.Series | None = None
        self.train:   pd.Series | None = None
        self.val:     pd.Series | None = None
        self.test:    pd.Series | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> "MarketData":
        """Download (or restore from cache) prices, compute returns, split.

        Returns
        -------
        self
            Enables method chaining: ``MarketData(...).load().summary()``.
        """
        self.prices  = self._download()
        self.returns = self._compute_returns(self.prices)
        self.train, self.val, self.test = self._split(self.returns)
        return self

    def summary(self) -> None:
        """Print a one-line summary for each data split."""
        self._require_loaded()
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            print(
                f"  {name:<6} {split.index[0].date()} → {split.index[-1].date()}"
                f"  ({len(split):>5} obs)"
                f"  ann.vol={self._ann_vol(split):.2%}"
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        start = pd.Timestamp(self.start).date().isoformat()
        end   = pd.Timestamp(self.end).date().isoformat()
        return _DOWNLOAD_DIR / f"{self.ticker}_{start}_{end}.parquet"

    def _load_cache(self, cache_file: Path) -> pd.Series | None:
        """Return cached prices if the file exists and covers the request window."""
        if not cache_file.exists():
            return None

        prices = pd.read_parquet(cache_file)["Close"].rename(self.ticker).sort_index()

        if prices.empty:
            return None

        # Accept the cache if its first bar is within 5 calendar days of start
        if prices.index[0] <= pd.Timestamp(self.start) + pd.Timedelta(days=5):
            return prices

        print(f"[{self.ticker}] Cache stale — re-downloading.")
        cache_file.unlink()
        return None

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self) -> pd.Series:
        """Return adjusted-close price series, using the local cache when fresh."""
        cache_file = self._cache_path()

        cached = self._load_cache(cache_file)
        if cached is not None:
            return cached

        print(f"[{self.ticker}] Downloading {self.start} → {self.end} via yfinance…")

        raw: pd.DataFrame = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,   # prices already adjusted; no 'Adj Close' column needed
            progress=False,
            threads=False,
        )

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {self.ticker!r} "
                f"between {self.start} and {self.end}. "
                "Check the ticker symbol and date range."
            )

        # yfinance ≥ 0.2.x returns a MultiIndex when downloading multiple
        # tickers; squeeze to a plain Series when we have only one.
        close: pd.Series = (
            raw["Close"].squeeze()
            if isinstance(raw.columns, pd.MultiIndex)
            else raw["Close"]
        )
        close = close.rename(self.ticker)

        # Drop timezone info so the index is tz-naive (consistent with the rest
        # of the pipeline and parquet round-trips).
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        close.sort_index(inplace=True)

        # Validate coverage
        earliest  = close.index[0]
        requested = pd.Timestamp(self.start)
        if earliest > requested + pd.Timedelta(days=5):
            raise ValueError(
                f"yfinance returned data starting {earliest.date()}, but you "
                f"requested start={self.start}.  The ticker may have been listed "
                f"later than that date, or the symbol is incorrect."
            )

        # Persist to cache
        close.to_frame(name="Close").to_parquet(cache_file)
        print(f"[{self.ticker}] Cached {len(close)} rows → {cache_file.name}")

        return close

    # ------------------------------------------------------------------
    # Returns & splits
    # ------------------------------------------------------------------

    def _compute_returns(self, prices: pd.Series) -> pd.Series:
        """Compute demeaned log returns."""
        log_returns = np.log(prices / prices.shift(1)).dropna()
        demeaned    = log_returns - log_returns.mean()
        demeaned.name = f"{self.ticker}_returns"
        return demeaned

    def _split(
        self, returns: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        train = returns[returns.index <  self.val_start]
        val   = returns[(returns.index >= self.val_start) & (returns.index < self.test_start)]
        test  = returns[returns.index >= self.test_start]

        if train.empty:
            raise ValueError(
                f"Train split is empty. Ensure start ({self.start}) is well "
                f"before val_start ({self.val_start.date()})."
            )
        if val.empty:
            raise ValueError(
                f"Validation split is empty. Extend the date range or adjust "
                f"val_start ({self.val_start.date()}) / test_start ({self.test_start.date()})."
            )
        if test.empty:
            raise ValueError(
                f"Test split is empty. Extend the date range or adjust "
                f"test_start ({self.test_start.date()})."
            )

        return train, val, test

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self.returns is None:
            raise RuntimeError("Call .load() before accessing data splits.")

    @staticmethod
    def _ann_vol(returns: pd.Series) -> float:
        """Annualised volatility (assuming 252 trading days)."""
        return float(returns.std() * np.sqrt(252))