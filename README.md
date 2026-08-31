# deepgarch

A neural-network-parameterized GARCH(1,1) — a small MLP maps features to
`(ω, α, β)` each day, subject to a stationarity constraint, and the usual
GARCH recursion runs on top. Evaluated against a static `arch`-fitted
GARCH(1,1) baseline on SPY, natural gas (`NG=F`), and WTI crude (`CL=F`).

## Results (test split)

QLIKE and MSE(var) score against **Parkinson range variance**
(`(ln(high/low))² / (4 ln 2)`), not squared returns — squared returns are an
unbiased but ~5x noisier daily variance estimator. Violations/Kupiec still
test actual realized returns against the forecast distribution, unaffected
by the proxy choice.

| Market  | Model         | QLIKE  | Violations (α=1%) | Kupiec p |
|---------|---------------|-------:|-------------------:|---------:|
| SPY     | GARCHNet      | −8.810 | 6/289 (2.08%)       | 0.108    |
| SPY     | Static GARCH  | −8.955 | 4/289 (1.38%)       | 0.535    |
| Natgas  | GARCHNet      | −5.506 | 20/1400 (1.43%)     | 0.130    |
| Natgas  | Static GARCH  | −5.462 | 17/1400 (1.21%)     | 0.436    |
| Oil     | GARCHNet      | −5.986 | 5/194 (2.58%)       | 0.065    |
| Oil     | Static GARCH  | −6.037 | 6/194 (3.09%)       | 0.019    |

QLIKE and violation ranking were re-checked on the validation split before
being read off the test split (see `results/ablations/` below) — the model
selection itself was validation-based; test numbers above are the final,
single look.

## The parameterisation story (`results/ablations/`)

Natural-gas GARCHNet across five parameterisations of the same architecture:

| Variant | QLIKE | Violations | Kupiec p |
|---|---:|---:|---:|
| `natgas_old` — softmax-shared (α,β) | −5.385 | 40/1400 (2.86%) | 1.2e-8 |
| `natgas_reparam` — sigmoid(ρ,φ), raw `ω` | −3.149 | 38/1400 (2.71%) | 1.0e-7 |
| `natgas_var_targeting` — `ω` reparameterised via unconditional variance target | −5.507 | 21/1400 (1.50%) | 0.080 |
| **`results/natgas`** — final (adds the scale-corrected tail-quantile diagnostic, no model change) | **−5.506** | **20/1400 (1.43%)** | **0.130** |
| `natgas_ablate` — level head disabled, `σ̄²` pinned to train-set unconditional variance | −5.463 | 17/1400 (1.21%) | 0.436 |

Decomposing `ω = (1-ρ)·σ̄²` and driving `σ̄²` from the unconditional variance
(`var_targeting`) is what fixed the reparameterisation — raw `ω` drifted and
wrecked QLIKE. The conditional level head (letting `σ̄²` move with features
rather than pinning it at the training-set unconditional variance,
`natgas_ablate`) buys close to nothing on QLIKE and, on the test split,
slightly hurts calibration — most of the gain is the GARCH(1,1) recursion
plus variance-targeting itself, not the conditional level. On oil the level
head is a wash on QLIKE (`results/oil` −5.986 vs `results/ablations/oil_ablate`
−5.982, within noise) but `oil_ablate`'s violation rate is worse
(3.09% vs 2.58%), same direction as natgas.

`natgas_old` / `natgas_reparam` / `natgas_var_targeting` are frozen
artifacts from earlier revisions of `ConditionalGARCHNet` — the constraint
math changed commit to commit, so they aren't reproducible via
`configs/natgas.yaml` on the current code. `natgas_ablate` / `oil_ablate`
share today's code and *are* reproducible, via `configs/*_ablate.yaml`.

## Install

```bash
conda env create -f environment.yml && conda activate deepgarch
# or: pip install -e .
```

## Run

```bash
python run.py --config configs/spy.yaml
python run.py --config configs/natgas.yaml
python run.py --config configs/oil.yaml
python run.py --config configs/spy_ablate.yaml
python run.py --config configs/natgas_ablate.yaml
python run.py --config configs/oil_ablate.yaml
```

Each run writes plots, `comparison_metrics.json`, `parameter_path.csv`,
`loss_series.csv` (per-observation QLIKE/MSE for every model),
`significance.json` (Diebold-Mariano vs the static-GARCH benchmark + the
Model Confidence Set), and a `*_model.pt` checkpoint to `output.dir` in the
config (`results/<market>` or `results/ablations/<variant>`).

### Seed sweep

`sweep.py` is a thin wrapper that re-runs configs across seeds, changing only
the seed and the output directory:

```bash
python sweep.py                                   # spy/natgas/oil x seeds 0-9
python sweep.py --seeds 0 1 2 --configs configs/natgas.yaml
```

Per-seed results land in `results/sweep/<market>/seed_<n>/`; aggregated
metrics (long form) plus across-seed mean/std go to `results/sweep/summary.csv`.
