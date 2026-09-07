"""Regenerate results/vmax_sweep.png from the committed sweep artifacts.

Two panels, both vs v_max, across the three markets on the test split:
  A. GARCHNet QLIKE minus the StaticGARCH benchmark (mean +/- seed sd) --
     negative = beats the benchmark. Shows conditioning helps only for
     natural gas at v_max = 1.
  B. GARCHNet seed-to-seed QLIKE sd (log scale). Shows the natural-gas
     level head becoming a variance bomb as the bound loosens.

Reads results/sweep_test_vmax/vmax_<v>/<market>/seed_*/comparison_metrics.json.
No training; just aggregation + matplotlib.
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = Path("results/sweep_test_vmax")
VMAX = [1.0, 2.0, 3.0, 4.0, 5.0]
MARKETS = {"nat_gas": "natural gas", "spy": "SPY", "oil": "oil"}
COLOR = {"nat_gas": "#c2410c", "spy": "#1d4ed8", "oil": "#047857"}


def series(market: str):
    """Return (delta_mean, delta_sd, seed_sd) arrays over VMAX for one market."""
    d_mean, d_sd, s_sd = [], [], []
    for v in VMAX:
        files = sorted(glob.glob(str(SWEEP / f"vmax_{v}" / market / "seed_*" / "comparison_metrics.json")))
        if not files:
            d_mean.append(np.nan); d_sd.append(np.nan); s_sd.append(np.nan); continue
        gn, bench = [], []
        for f in files:
            m = json.loads(Path(f).read_text())
            gkey = next(k for k in m if "GARCHNet" in k)
            gn.append(m[gkey]["qlike"])
            bench.append(m["StaticGARCH"]["qlike"])
        gn, bench = np.array(gn), np.array(bench)
        delta = gn - bench
        d_mean.append(delta.mean())
        d_sd.append(delta.std(ddof=1))
        s_sd.append(gn.std(ddof=1))
    return np.array(d_mean), np.array(d_sd), np.array(s_sd)


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    for mk, label in MARKETS.items():
        dm, dsd, ssd = series(mk)
        c = COLOR[mk]
        axA.errorbar(VMAX, dm, yerr=dsd, marker="o", ms=5, lw=1.6, capsize=3, color=c, label=label)
        axB.plot(VMAX, ssd, marker="o", ms=5, lw=1.6, color=c, label=label)

    axA.axhline(0, color="black", lw=0.8, ls="--")
    axA.set_xlabel("v_max")
    axA.set_ylabel("QLIKE(GARCHNet) − QLIKE(StaticGARCH)")
    axA.set_title("A. Skill vs benchmark  (negative = better; mean ± seed sd)")
    axA.set_xticks(VMAX)
    axA.legend(frameon=False)

    axB.set_yscale("log")
    axB.set_xlabel("v_max")
    axB.set_ylabel("GARCHNet seed-to-seed QLIKE sd")
    axB.set_title("B. Run-to-run stability  (10 seeds, log scale)")
    axB.set_xticks(VMAX)
    axB.legend(frameon=False)

    fig.suptitle("v_max sweep — 10 seeds, test split", y=1.02, fontsize=12)
    fig.tight_layout()
    out = Path("results/vmax_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
