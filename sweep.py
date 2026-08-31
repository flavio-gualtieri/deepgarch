"""Thin seed-sweep wrapper around run.py.

Runs each config across N seeds, changing only `config.seed` and
`config.output.dir` — every other setting is the YAML verbatim. Results
land in results/sweep/<market>/seed_<n>/, and per-run metrics are
aggregated (long form) into results/sweep/summary.csv.

    python sweep.py                                  # 3 main configs x seeds 0-9
    python sweep.py --seeds 0 1 2 --configs configs/natgas.yaml
"""

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from deepgarch.config import RunConfig
from run import run

MAIN_CONFIGS = ["configs/spy.yaml", "configs/natgas.yaml", "configs/oil.yaml"]
_METRIC_KEYS = ("qlike", "mse_variance")
_VAR_KEYS = ("n_obs", "n_violations", "violation_rate", "kupiec_pvalue")


def _rows_from_metrics(market: str, seed: int, path: Path):
    metrics = json.loads(path.read_text())
    for model, m in metrics.items():
        row = {"market": market, "seed": seed, "model": model}
        row.update({k: m.get(k) for k in _METRIC_KEYS})
        row.update({k: m.get("var", {}).get(k) for k in _VAR_KEYS})
        row["chris_p_cc"] = m.get("christoffersen", {}).get("p_cc")
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--configs", nargs="+", default=MAIN_CONFIGS)
    parser.add_argument("--out", default="results/sweep")
    args = parser.parse_args()

    out_root = Path(args.out)
    failures = []
    for cfg_path in args.configs:
        for seed in args.seeds:
            config = RunConfig.from_yaml(cfg_path)
            config.seed = seed
            config.output.dir = str(out_root / config.output.market / f"seed_{seed}")
            banner = f"{cfg_path}  seed={seed}  ->  {config.output.dir}"
            print(f"\n{'=' * len(banner)}\n{banner}\n{'=' * len(banner)}", flush=True)
            try:
                run(config)
            except Exception:
                failures.append((cfg_path, seed))
                print(f"!! FAILED {cfg_path} seed={seed}\n{traceback.format_exc()}", flush=True)

    rows = []
    for cfg_path in args.configs:
        market = RunConfig.from_yaml(cfg_path).output.market
        for seed in args.seeds:
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

    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
    main()
