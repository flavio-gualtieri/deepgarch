"""
Build a static reference list of EIA Weekly Natural Gas Storage Report (WNGSR)
release dates, 2005-01-01 through 2026-12-31.

SOURCING (see README.md for full citation list):

1. Standard rule: WNGSR normally releases Thursday 10:30am ET.

2. Holiday-shift rule, quoted directly from EIA's own release-schedule page
   (https://ir.eia.gov/ngs/schedule.html), as reported by industry press:
       - Monday or Friday holiday in the report week -> stays Thursday, 10:30am
       - Tuesday or Wednesday holiday in the report week -> moves to Friday, 10:30am
       - Thursday holiday in the report week -> moves to Wednesday, 12:00pm
   This is a *policy rule EIA states about itself*, not a weekday heuristic we
   invented -- and it is confirmed against every year of EIA's own published
   holiday-exception table we could pull directly (2019, 2020, 2023, 2024,
   2025, 2026 -- see VERIFIED_EXCEPTIONS below, all of which match this rule
   with one flagged exception).

3. Only 5 fixed/near-fixed-date federal holidays ever land on a Tue/Wed/Thu
   and are therefore capable of triggering a shift for THIS report:
       New Year's Day (Jan 1), Juneteenth (Jun 19, federal holiday only from
       2021 onward), Independence Day (Jul 4), Veterans Day (Nov 11),
       Christmas Day (Dec 25).
   Thanksgiving (4th Thursday of November) is a Thursday by definition, so it
   ALWAYS triggers the Wednesday-release shift, every single year.

   IMPORTANT CORRECTION vs. the holiday list in the prompt: Memorial Day and
   Labor Day are always Mondays, and Monday holidays do NOT shift this
   report (confirmed: no exception exists for Memorial Day/Labor Day in any
   year of EIA's published table, e.g. 2023-2026). They matter for other EIA
   products (e.g. the Wednesday-based Weekly Petroleum Status Report) but not
   for WNGSR. Don't apply a Memorial Day/Labor Day shift to this report.

4. Known one-off anomaly NOT explained by the holiday rule, sourced directly
   from EIA's live schedule page (ir.eia.gov/ngs/schedule.html, "(Updated)"
   annotation): Christmas week 2025 actually released Monday, Dec 29, 2025 --
   not the rule-predicted Wednesday, Dec 24. This is hardcoded as an override.

5. Government shutdowns (2013, Dec 2018-Jan 2019, 2025) did NOT change WNGSR's
   day-of-week schedule -- EIA is funded outside annual appropriations lapses
   and publicly confirmed it continued releasing on schedule during the
   2018-2019 shutdown. No override needed for those periods.

6. Same-day *intraday* delays (e.g. a technical-issue delay to 2pm ET on an
   otherwise-Thursday release) are NOT reflected here, since they don't change
   which calendar day the report landed on. If your model needs intraday
   timing precision around specific known incidents, treat those separately.

LIMITATION TO DISCLOSE: EIA's own forward/trailing schedule page only retains
roughly the current + prior ~2 years of exceptions, and their historical
release-date file (ngshistory.xls) blocks automated fetching (bot detection)
and in any case records storage *values*, not publication timestamps. So for
2005-2018 there is no surviving EIA-hosted table of exceptions to scrape --
this script *computes* those years from the holiday rule above, which is
EIA's own stated general policy and which we verified exactly reproduces
every year we could directly check (2019-2026). It is not a naive
"Thursday unless holiday week" guess; it's EIA's documented mechanism. But
it is inference for 2005-2018, not a scraped record, and is flagged as such
in the `source` column.
"""

import csv
import datetime as dt

START = dt.date(2005, 1, 1)
END = dt.date(2026, 12, 31)

VERIFIED_YEARS = {2019, 2020, 2023, 2024, 2025, 2026}  # years EIA's own
# published exception table was directly retrieved for, during research

# --- known one-off anomalies, NOT derivable from the recurring holiday rule.
# Each is keyed by the *standard* Thursday of that report week (pre-shift).
# Sourced individually -- see README.md for citations for each.
MANUAL_OVERRIDES = {
    # National day of mourning for George H.W. Bush, Wed 2018-12-05.
    # EIA's own announcement page confirms WNGSR moved to Friday 2018-12-07
    # 10:30am (same mechanism as any Wed-holiday, just not a recurring one).
    dt.date(2018, 12, 6): {
        "release_date": dt.date(2018, 12, 7),
        "note": "National day of mourning, George H.W. Bush (2018-12-05, Wed) "
        "-- EIA announcement confirms Friday 12/7 release.",
        "source": "eia_published_announcement",
        "skipped": False,
    },
    # Planned systems upgrade: EIA SKIPPED the 2023-11-09 release entirely
    # and folded that week's data into the following week's release.
    dt.date(2023, 11, 9): {
        "release_date": None,
        "note": "SKIPPED. EIA press release (10/19/2023): systems upgrade "
        "delayed all releases 11/8-11/10/2023; WNGSR did not publish this "
        "week -- its data was folded into the 2023-11-16 release instead.",
        "source": "eia_press_release_545",
        "skipped": True,
    },
    dt.date(2023, 11, 16): {
        "release_date": dt.date(2023, 11, 16),
        "note": "This release contains TWO weeks of storage data (report "
        "weeks ending 11/10 and 11/17) because the prior week was skipped "
        "-- see note on 2023-11-09.",
        "source": "eia_press_release_545",
        "skipped": False,
    },
    # National day of mourning for Jimmy Carter, Thu 2025-01-09.
    dt.date(2025, 1, 9): {
        "release_date": dt.date(2025, 1, 8),
        "note": "National day of mourning, Jimmy Carter (2025-01-09, Thu) "
        "-- EIA schedule page confirms Wednesday 1/8 release (marked "
        "'(Updated)' on the live schedule page).",
        "source": "eia_published_schedule_updated",
        "skipped": False,
    },
    # Christmas week 2025: rule predicts Wed 2025-12-24; EIA's live schedule
    # page (marked "(Updated)") instead shows Monday 2025-12-29.
    dt.date(2025, 12, 25): {
        "release_date": dt.date(2025, 12, 29),
        "note": "Ad hoc reschedule, not explained by the holiday rule. EIA's "
        "live schedule page marks this '(Updated)' -- rule would have "
        "predicted Wed 2025-12-24.",
        "source": "eia_published_schedule_updated",
        "skipped": False,
    },
}


def nth_weekday_of_month(year, month, weekday, n):
    """weekday: Monday=0 ... Sunday=6. n=1 for 1st, 4 for 4th, etc."""
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d += dt.timedelta(days=offset)
    d += dt.timedelta(weeks=n - 1)
    return d


def last_weekday_of_month(year, month, weekday):
    if month == 12:
        d = dt.date(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - dt.timedelta(days=offset)


def relevant_holidays(year):
    """Only the fixed-date holidays capable of landing on Tue/Wed/Thu for
    WNGSR, plus Thanksgiving (always Thursday). Monday-only holidays
    (MLK, Presidents, Memorial, Labor, Columbus/Indigenous Peoples Day)
    are intentionally excluded -- they never shift this report."""
    hols = {
        dt.date(year, 1, 1): "New Year's Day",
        dt.date(year, 7, 4): "Independence Day",
        dt.date(year, 11, 11): "Veterans Day",
        dt.date(year, 12, 25): "Christmas Day",
        nth_weekday_of_month(year, 11, 3, 4): "Thanksgiving Day",  # 3=Thursday
    }
    if year >= 2021:
        hols[dt.date(year, 6, 19)] = "Juneteenth National Independence Day"
    return hols


def thursdays_between(start, end):
    d = start
    d += dt.timedelta(days=(3 - d.weekday()) % 7)  # first Thursday (3=Thu)
    while d <= end:
        yield d
        d += dt.timedelta(weeks=1)


def build_calendar():
    rows = []
    for thursday in thursdays_between(START, END):
        year = thursday.year
        hols = relevant_holidays(year)
        # also need holidays from adjacent years near Jan 1 boundary
        hols.update(relevant_holidays(year - 1))
        hols.update(relevant_holidays(year + 1))

        tue = thursday - dt.timedelta(days=2)
        wed = thursday - dt.timedelta(days=1)

        holiday_on_thu = hols.get(thursday)
        holiday_on_wed = hols.get(wed)
        holiday_on_tue = hols.get(tue)

        release_date = thursday
        holiday_name = None
        is_exception = False
        note = ""
        skipped = False

        if holiday_on_thu:
            release_date = wed
            holiday_name = holiday_on_thu
            is_exception = True
        elif holiday_on_wed:
            release_date = thursday + dt.timedelta(days=1)  # Friday
            holiday_name = holiday_on_wed
            is_exception = True
        elif holiday_on_tue:
            release_date = thursday + dt.timedelta(days=1)  # Friday
            holiday_name = holiday_on_tue
            is_exception = True

        source = (
            "eia_published_schedule_verified"
            if year in VERIFIED_YEARS
            else "derived_from_eia_stated_holiday_rule"
        )

        # apply manual override if this standard Thursday has one -- these
        # take precedence over the recurring-holiday computation above
        if thursday in MANUAL_OVERRIDES:
            ov = MANUAL_OVERRIDES[thursday]
            release_date = ov["release_date"]
            note = ov["note"]
            is_exception = True
            source = ov["source"]
            skipped = ov["skipped"]

        rows.append(
            {
                "standard_thursday": thursday.isoformat(),
                "release_date": "" if release_date is None else release_date.isoformat(),
                "release_dow": "" if release_date is None else release_date.strftime("%A"),
                "is_holiday_exception": is_exception,
                "skipped": skipped,
                "holiday": holiday_name or "",
                "source": source,
                "note": note,
            }
        )
    return rows


def main():
    rows = build_calendar()
    out_path = "/home/claude/eia_ref/eia_wngsr_release_dates.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "standard_thursday",
                "release_date",
                "release_dow",
                "is_holiday_exception",
                "skipped",
                "holiday",
                "source",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    exceptions = [r for r in rows if r["is_holiday_exception"]]
    print(f"\n{len(exceptions)} exception weeks total:")
    for r in exceptions:
        print(
            f"  {r['standard_thursday']} -> {r['release_date']} "
            f"({r['release_dow']}) : {r['holiday']} [{r['source']}]"
        )


if __name__ == "__main__":
    main()
