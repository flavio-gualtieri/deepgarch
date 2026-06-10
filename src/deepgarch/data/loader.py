# src/deepgarch/data/loader.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests

_DEFAULT_VAL_START  = "2018-01-01"
_DEFAULT_TEST_START = "2021-01-01"
_DOWNLOAD_DIR = Path("/Users/qp252676/Desktop/deepgarch/src/deepgarch/data/downloaded")

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

    
    def _cache_path(self) -> Path:
        _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        ticker = self.ticker.upper()
        start = pd.Timestamp(self.start).date().isoformat()
        end = pd.Timestamp(self.end).date().isoformat()

        return _DOWNLOAD_DIR / f"{ticker}_{start}_{end}.parquet"


    def _download(self) -> pd.Series:
        cache_file = self._cache_path()
        if cache_file.exists():
            prices = pd.read_parquet(cache_file)["Close"]
            prices.name = self.ticker
            prices.sort_index(inplace=True)

            if not prices.empty and prices.index[0] <= pd.Timestamp(self.start) + pd.Timedelta(days=5):
                return prices
            else:
                print(f"Cache miss or stale for {self.ticker}, re-downloading...")
                cache_file.unlink()

        api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")

        if not api_key:
            raise RuntimeError(
                "No API key found. Set the MASSIVE_API_KEY (or POLYGON_API_KEY) "
                "environment variable."
            )

        url = (
            f"https://api.massive.com/v2/aggs/ticker/{self.ticker}"
            f"/range/1/day/{self.start}/{self.end}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                raise RuntimeError(
                    "Massive API rate limit hit. Wait and try again."
                ) from exc
            raise

        data = resp.json()

        if data.get("status") == "ERROR":
            raise RuntimeError(
                f"Massive API error for {self.ticker!r}: {data.get('error', data)}"
            )

        results = data.get("results") or []
        print(f"API returned {len(results)} rows; first={results[0] if results else None}")

        if not results:
            raise ValueError(
                f"Massive returned no data for {self.ticker!r} "
                f"between {self.start} and {self.end}."
            )

        index = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True).tz_convert(None)
        prices = pd.Series(
            [r["c"] for r in results],
            index=index,
            name=self.ticker,
            dtype=float,
        )
        prices.sort_index(inplace=True)

        # ✅ Validate coverage BEFORE caching
        earliest = prices.index[0]
        requested = pd.Timestamp(self.start)
        if earliest > requested + pd.Timedelta(days=5):
            raise ValueError(
                f"Massive returned data starting {earliest.date()}, but you requested "
                f"start={self.start}. Your API plan may not have access to data before "
                f"{earliest.date()}. Either adjust `start` or upgrade your data subscription."
            )

        prices.to_frame(name="Close").to_parquet(cache_file)
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
        val = returns[(returns.index >= self.val_start) & (returns.index < self.test_start)]
        test = returns[returns.index >= self.test_start]
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