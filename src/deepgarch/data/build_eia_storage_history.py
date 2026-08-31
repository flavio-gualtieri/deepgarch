"""
Build a clean weekly EIA natural gas storage history, aligned to the actual
public release date, for use as an exogenous feature in FeaturePipeline.

SOURCE: https://ir.eia.gov/ngs/ngshistory.xls -- EIA's own maintained archive
of the Weekly Natural Gas Storage Report (WNGSR), Total Lower 48. Two sheets:
  - html_report_history: weekly storage LEVEL (Bcf), 2010-01-01 -> present.
  - weekly_net_changes:   weekly net CHANGE (Bcf), same range.

ALIGNMENT: ngshistory.xls indexes rows by "week ending" (the Friday the
survey covers), not by the date the number became public. The report for
week_ending W is normally released on Thursday W+6 days -- confirmed against
the live wngsr.csv snapshot ("Released July 30, 2026 ... Week Ending July
24, 2026", a 6-day gap) and checked against every one of the 865 rows here,
which all match src/deepgarch/data/eia_wngsr_release_dates.csv's
standard_thursday column exactly. That calendar file (see
build_eia_release_calendar.py) already accounts for holiday shifts and the
one known publication skip (week_ending 2023-11-03 was folded into the
2023-11-16 release), so joining on standard_thursday and taking its
release_date gives the true date each week's figures became public -- not a
generic fixed-day-count guess.

DERIVED FEATURE -- storage_vs_5yr_avg_bcf: deviation from the trailing
5-year seasonal average, which is closer to what actually moves the market
than the raw level (which is dominated by the calendar season). Computed
positionally (52 rows back per prior year) rather than by calendar date,
since this is a strictly-weekly series -- 52*k rows back lands within a few
days of the same calendar week k years earlier, which is simpler and no
less accurate than exact date matching for a ~7-day-spaced series. Requires
5 full prior years of history, so it's only populated from ~2015 onward
even though the raw series starts in 2010.
"""

import urllib.request
from pathlib import Path

import pandas as pd

_XLS_URL = "https://ir.eia.gov/ngs/ngshistory.xls"
_CALENDAR_PATH = Path(__file__).parent / "eia_wngsr_release_dates.csv"
_OUT_PATH = Path(__file__).parent / "eia_storage_history.csv"
_RAW_CACHE = Path(__file__).parent / "downloaded" / "ngshistory.xls"

_WEEKS_PER_YEAR = 52
_YEARS_FOR_AVG = 5


def _download(force: bool = False) -> Path:
    _RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if force or not _RAW_CACHE.exists():
        req = urllib.request.Request(_XLS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            _RAW_CACHE.write_bytes(resp.read())
    return _RAW_CACHE


def _load_sheets(xls_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(xls_path)
    level = pd.read_excel(xls, sheet_name="html_report_history", header=6)
    change = pd.read_excel(xls, sheet_name="weekly_net_changes", header=6)
    change.columns = [
        "Week ending", "Source", "East Region", "Midwest Region", "Mountain Region",
        "Pacific Region", "South Central Region", "Salt", "NonSalt", "Total Lower 48",
    ]
    return level[["Week ending", "Total Lower 48"]], change[["Week ending", "Total Lower 48"]]


def _actual_release_date(standard_thursday: pd.Series, calendar: pd.DataFrame) -> pd.Series:
    """Map each report week to the date its data actually became public.

    Handles the one known skip (2023-11-09) by falling forward to the next
    calendar row that actually published -- that week's figures went out
    folded into the following release, not on its own (blank) date.
    """
    cal = calendar.sort_values("standard_thursday").reset_index(drop=True)
    cal["release_date"] = cal["release_date"].bfill()
    lookup = cal.set_index("standard_thursday")["release_date"]
    return standard_thursday.map(lookup)


def build() -> pd.DataFrame:
    xls_path = _download()
    level, change = _load_sheets(xls_path)

    level = level.rename(columns={"Total Lower 48": "storage_level_bcf"})
    change = change.rename(columns={"Total Lower 48": "net_change_bcf"})

    df = level.merge(change, on="Week ending", how="inner").sort_values("Week ending").reset_index(drop=True)
    df["standard_thursday"] = df["Week ending"] + pd.Timedelta(days=6)

    calendar = pd.read_csv(_CALENDAR_PATH, parse_dates=["standard_thursday", "release_date"])
    df["date"] = _actual_release_date(df["standard_thursday"], calendar)

    if df["date"].isna().any():
        missing = df[df["date"].isna()]["Week ending"].tolist()
        raise ValueError(f"No release date resolved for week(s): {missing}")

    # Trailing 5-year seasonal average, computed positionally (see module docstring).
    lagged = pd.concat(
        [df["storage_level_bcf"].shift(_WEEKS_PER_YEAR * k) for k in range(1, _YEARS_FOR_AVG + 1)],
        axis=1,
    )
    trailing_5yr_avg = lagged.mean(axis=1, skipna=False)
    df["storage_vs_5yr_avg_bcf"] = df["storage_level_bcf"] - trailing_5yr_avg

    out = df[["date", "net_change_bcf", "storage_vs_5yr_avg_bcf"]].dropna(subset=["net_change_bcf"])
    out = out.sort_values("date")

    # The one folded-release week (see module docstring) leaves two report
    # weeks sharing a single public date. Collapse to one row per date --
    # downstream joins assume a unique date index. Sum the net changes (both
    # became known simultaneously); keep the later week's seasonal deviation
    # (the standing after both are accounted for).
    out = out.groupby("date", as_index=False).agg(
        net_change_bcf=("net_change_bcf", "sum"),
        storage_vs_5yr_avg_bcf=("storage_vs_5yr_avg_bcf", "last"),
    )
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    out = build()
    out.to_csv(_OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows -> {_OUT_PATH}")
    print(f"date range: {out['date'].min().date()} -> {out['date'].max().date()}")
    n_seasonal = out["storage_vs_5yr_avg_bcf"].notna().sum()
    print(f"storage_vs_5yr_avg_bcf populated for {n_seasonal}/{len(out)} rows "
          f"(starts once {_YEARS_FOR_AVG} full prior years exist)")


if __name__ == "__main__":
    main()
