"""Thin seed-sweep wrapper around run.py.

Runs each config across N seeds, changing only `config.seed` and
`config.output.dir` — every other setting is the YAML verbatim. Per-seed
output lands in results/sweep/<market>/seed_<n>/; the sweep is then
aggregated into

    results/sweep/summary.csv        long-form per-(market, seed, model) metrics
    results/sweep/significance.json  DM + Model Confidence Set across all seeds
                                     (ensemble = seed-averaged loss series, plus
                                     the per-seed distribution)

    python sweep.py                                  # 3 main configs x seeds 0-9
    python sweep.py --seeds 0 1 2 --configs configs/natgas.yaml
    python sweep.py --aggregate-only                 # rebuild the two files, no training
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from deepgarch.config import RunConfig
from deepgarch.eval import diebold_mariano, model_confidence_set
from run import run

MAIN_CONFIGS = ["configs/spy.yaml", "configs/natgas.yaml", "configs/oil.yaml"]
BENCHMARK = "StaticGARCH"
_METRIC_KEYS = ("qlike", "mse_variance")
_VAR_KEYS = ("n_obs", "n_violations", "violation_rate", "kupiec_pvalue")
_LOSSES = ("qlike", "mse_var")  # column suffixes in loss_series.csv


def _rows_from_metrics(market: str, seed: int, path: Path):
    metrics = json.loads(path.read_text())
    for model, m in metrics.items():
        row = {"market": market, "seed": seed, "model": model}
        row.update({k: m.get(k) for k in _METRIC_KEYS})
        row.update({k: m.get("var", {}).get(k) for k in _VAR_KEYS})
        row["chris_p_cc"] = m.get("christoffersen", {}).get("p_cc")
        yield row


def _load_loss_series(out_root: Path, market: str, seeds) -> dict[int, pd.DataFrame]:
    frames = {}
    for seed in seeds:
        p = out_root / market / f"seed_{seed}" / "loss_series.csv"
        if p.exists():
            frames[seed] = pd.read_csv(p, index_col=0)
    return frames


def _aggregate_significance(out_root: Path, market: str, seeds) -> dict | None:
    """DM + MCS over the sweep: `ensemble` runs the test once on the loss
    series averaged across seeds; `per_seed` is the distribution of the same
    test run separately per seed. Baselines are seed-independent, so their
    per-seed entries are constant by construction."""
    frames = _load_loss_series(out_root, market, seeds)
    if len(frames) < 2:
        return None
    seed_ids = sorted(frames)
    cols = next(iter(frames.values())).columns
    models = list(dict.fromkeys(c.rsplit(" | ", 1)[0] for c in cols))
    if BENCHMARK not in models:
        return None

    out = {"benchmark": BENCHMARK, "n_seeds": len(seed_ids), "seeds": seed_ids}

    for loss in _LOSSES:
        def series(seed, model):
            return frames[seed][f"{model} | {loss}"].to_numpy()

        ensemble = {m: np.mean([series(s, m) for s in seed_ids], axis=0) for m in models}

        dm = {}
        for m in models:
            if m == BENCHMARK:
                continue
            per_seed = [diebold_mariano(series(s, m), series(s, BENCHMARK)) for s in seed_ids]
            ps = np.array([r["p_value"] for r in per_seed])
            dm[m] = {
                "ensemble": diebold_mariano(ensemble[m], ensemble[BENCHMARK]),
                "per_seed": {
                    "mean_p": float(ps.mean()),
                    "median_p": float(np.median(ps)),
                    "min_p": float(ps.min()),
                    "max_p": float(ps.max()),
                    "frac_sig_0.05": float((ps < 0.05).mean()),
                    "mean_dm_stat": float(np.mean([r["dm_stat"] for r in per_seed])),
                    "mean_loss_diff": float(np.mean([r["mean_loss_diff"] for r in per_seed])),
                    "p_values": ps.tolist(),
                },
            }

        try:
            mcs_ensemble = model_confidence_set({m: ensemble[m] for m in models})
        except Exception as exc:  # MCS bootstrap can fail on a degenerate loss series
            mcs_ensemble = {"error": repr(exc)}

        included = {m: 0 for m in models}
        pvalues = {m: [] for m in models}
        n_ok = 0
        for s in seed_ids:
            try:
                r = model_confidence_set({m: series(s, m) for m in models})
            except Exception:
                continue
            n_ok += 1
            for m in r["included"]:
                included[m] += 1
            for m, pv in r["pvalues"].items():
                pvalues[m].append(float(pv))

        out[loss] = {
            "dm_vs_benchmark": dm,
            "mcs": {
                "ensemble": mcs_ensemble,
                "per_seed": {
                    "n_seeds_ok": n_ok,
                    "in_mcs_count": included,
                    "mean_pvalue": {
                        m: (float(np.mean(pvalues[m])) if pvalues[m] else None) for m in models
                    },
                },
            },
        }
    return out


def aggregate(out_root: Path, configs, seeds) -> None:
    markets = list(dict.fromkeys(RunConfig.from_yaml(c).output.market for c in configs))

    rows = []
    for market in markets:
        for seed in seeds:
            mpath = out_root / market / f"seed_{seed}" / "comparison_metrics.json"
            if mpath.exists():
                rows.extend(_rows_from_metrics(market, seed, mpath))
    if rows:
        summary = pd.DataFrame(rows).sort_values(["market", "model", "seed"])
        summary_path = out_root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nsaved -> {summary_path}  ({len(summary)} rows)")
        agg = (
            summary.groupby(["market", "model"])[["qlike", "violation_rate", "kupiec_pvalue"]]
            .agg(["mean", "std"])
            .round(4)
        )
        print("\nacross-seed mean / std:\n")
        print(agg.to_string())

    significance = {}
    for market in markets:
        sig = _aggregate_significance(out_root, market, seeds)
        if sig is not None:
            significance[market] = sig
    if significance:
        sig_path = out_root / "significance.json"
        sig_path.write_text(json.dumps(significance, indent=2, default=float))
        print(f"\nsaved -> {sig_path}")
        for market, sig in significance.items():
            gn = f"{market} GARCHNet"
            q = sig["qlike"]
            dm = q["dm_vs_benchmark"].get(gn, {})
            incl = q["mcs"]["per_seed"]["in_mcs_count"]
            n = sig["n_seeds"]
            print(
                f"  {market:8} GARCHNet vs {BENCHMARK}: DM p ensemble="
                f"{dm.get('ensemble', {}).get('p_value', float('nan')):.4f}, "
                f"per-seed sig@0.05 {dm.get('per_seed', {}).get('frac_sig_0.05', 0) * n:.0f}/{n}; "
                f"in QLIKE MCS {incl.get(gn, 0)}/{n} seeds"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--v-max", type=float, nargs="+", default=[None])
    parser.add_argument("--configs", nargs="+", default=MAIN_CONFIGS)
    parser.add_argument("--out", default="results/sweep")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip training; rebuild summary.csv + significance.json from existing seed dirs",
    )
    args = parser.parse_args()
    out_root = Path(args.out)

    def vroot_for(v):
        return out_root if v is None else out_root / f"vmax_{v}"

    failures = []
    if not args.aggregate_only:
        for cfg_path in args.configs:
            for v in args.v_max:
                vroot = vroot_for(v)
                for seed in args.seeds:
                    config = RunConfig.from_yaml(cfg_path)
                    config.seed = seed
                    if v is not None:
                        config.model.v_max = v
                    config.output.dir = str(vroot / config.output.market / f"seed_{seed}")
                    banner = f"{cfg_path}  seed={seed}  ->  {config.output.dir}"
                    print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}", flush=True)
                    try:
                        run(config)
                    except Exception:
                        failures.append((cfg_path, seed))
                        print(f"!! FAILED {cfg_path} seed={seed}\n{traceback.format_exc()}", flush=True)

    for v in args.v_max:
        aggregate(vroot_for(v), args.configs, args.seeds)

    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
    main()
