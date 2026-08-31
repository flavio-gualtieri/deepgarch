# deepgarch

**Does conditioning GARCH parameters on market state actually improve volatility forecasts?**

GARCH(1,1) holds `(ω, α, β)` constant for all time. This project makes them a
function of observable state — a small MLP maps lagged returns, realised
volatility, volume and exogenous series to daily `(ω, α, β)` — and trains the
whole thing end to end by backpropagating through the variance recursion.
Benchmarked against static GARCH, GJR-GARCH, EGARCH and EWMA on SPY, natural
gas (`NG=F`) and WTI crude (`CL=F`).

**Answer: mostly no, and the failure mode is more interesting than the wins.**
Across 10 seeds the conditional model is statistically indistinguishable from
the GARCH family on equities, and on natural gas it under-forecasts variance by
~2.6×. An ablation localises what performance there is to the recursion and
variance targeting, not the learned features.

---

## Model

The MLP emits three unbounded reals per day, mapped to valid GARCH parameters
in `ConditionalGARCHNet._constrain_path`:

```
ρ  = max_persistence · sigmoid(ρ_raw)          persistence (α+β), capped < 1
α  = ρ · s_max · sigmoid(φ_raw)                 split of ρ into the shock term
β  = ρ − α
σ̄² = exp(v₀ + v_max · tanh(v_raw / v_max))      long-run variance level
ω  = (1 − ρ) · σ̄²                               variance targeting
```

This enforces stationarity by construction: `ω > 0`, `α, β > 0`,
`α + β < max_persistence`. The variance recursion `h_t = ω_t + α_t r²_{t−1} +
β_t h_{t−1}` then runs on top, trained on Gaussian NLL.

Every feature is `.shift(1)`-lagged in `features/pipeline.py`. The one
deliberate exception is `is_eia_day`, shifted `−1`: the recursion consumes row
`t−1`'s parameters to produce `σ²_t`, so the EIA storage-release flag must sit
on the row *before* the release for the model to anticipate it.

## Results

10 seeds per market, mean ± sd. Baselines are deterministic `arch` fits, so
they carry no seed variation. QLIKE and MSE score against **Parkinson range
variance**, `(ln(high/low))² / (4 ln 2)` — an unbiased but ~5× less noisy daily
variance proxy than squared returns. Violations and Kupiec test realised
returns against the forecast distribution and are unaffected by the proxy.

| Market | Model | QLIKE (mean ± sd) | Violations (α=1%) | Kupiec p |
|---|---|---:|---:|---:|
| **SPY** (n=313) | GARCHNet | −9.0493 ± 0.0171 | 1.28% | 0.636 |
| | EGARCH | −9.0683 | 1.92% | 0.148 |
| | GJR-GARCH | −9.0224 | 1.28% | 0.636 |
| | EWMA | −8.9856 | 1.28% | 0.636 |
| | Static GARCH | −8.9689 | 1.28% | 0.636 |
| **Natgas** (n=1423) | GARCHNet | −5.4126 ± 0.0389 | 1.56% | 0.067 |
| | EGARCH | −5.5039 | 1.12% | 0.644 |
| | Static GARCH | −5.4812 | 1.19% | 0.474 |
| | GJR-GARCH | −5.4727 | 1.34% | 0.227 |
| | EWMA | −5.4554 | 1.05% | 0.839 |
| **Oil** (n=229) | Static GARCH | −6.0339 | 2.62% | 0.040 |
| | GARCHNet | −6.0038 ± 0.0046 | 3.14% | 0.010 |
| | EWMA | −5.9547 | 2.62% | 0.040 |
| | GJR-GARCH | −5.8246 | 3.93% | 0.001 |
| | EGARCH | −5.5915 | 3.93% | 0.001 |

Full per-seed numbers in `results/sweep/summary.csv`.

**SPY.** The conditional model beats static GARCH, GJR and EWMA but loses to
EGARCH on 9 of 10 seeds. Seed-to-seed sd is 0.0171 against a 0.019 QLIKE gap to
EGARCH — the run-to-run noise is the same size as the effect. The Model
Confidence Set (α=0.1, `results/spy/significance.json`) retains four of five
models, excluding only static GARCH. The correct reading is *competitive with
the GARCH family, not separable from it*.

**Natural gas.** The clearest negative, and the one worth reading carefully.
GARCHNet has the **lowest variance MSE** of all five models (1.23e−5 vs 1.87e−5
for static GARCH) and simultaneously the **worst QLIKE** and a rejected VaR
backtest (Kupiec p = 0.002, Christoffersen p_cc = 0.008 on the reference seed).

The diagnostic that reconciles this is `mean_z2` — the mean of squared
standardised residuals `(r_t/σ_t)²`, which should be 1 if the variance forecast
is correctly scaled. GARCHNet gives **2.61**; every baseline sits at 1.11–1.24.
The model under-forecasts variance by a factor of ~2.6.

That single number explains the whole pattern. MSE is symmetric and rewards
tracking the *shape* of the variance path, which the model does well. QLIKE and
VaR both punish under-prediction asymmetrically, so a scale error of this size
destroys both. The `scale_corrected_quant` diagnostic (−1.66 vs a Gaussian
−2.33) confirms the tail *shape* is fine once the scale is divided out. So this
is a calibration failure of the level, not a fat-tail or regime problem —
violations don't even cluster (`p_ind` = 0.55).

**Oil.** Second on QLIKE behind static GARCH, ahead of EWMA, GJR and EGARCH,
with the tightest seed spread (sd 0.0046). Every model fails VaR coverage on
this window. Note the oil test split is short (229 days) and ends earlier than
the other two markets.

## Ablations (`results/ablations/`)

Disabling the conditional level head — pinning `σ̄²` to the training-set
unconditional variance, everything else identical — costs almost nothing:

| Market | Full model | Level head disabled |
|---|---:|---:|
| SPY | −9.0793 | −9.0580 |
| Natgas | −5.3532 | −5.4823 |
| Oil | −5.9959 | −6.0042 |

On natural gas and oil the ablated model is *better*. The performance lives in
the GARCH recursion and the variance-targeting parameterisation, not in letting
features move the variance level. (Single-seed, `seed: 42`.)

`natgas_no_eia` drops the EIA storage-release features: −5.5074 vs −5.3532,
i.e. the event features hurt on the test split.

Three frozen artifacts record how the parameterisation was fixed:

| Variant | QLIKE | Kupiec p |
|---|---:|---:|
| `natgas_old` — softmax-shared (α,β) | −5.385 | 1.2e−8 |
| `natgas_reparam` — sigmoid(ρ,φ), raw `ω` | −3.149 | 1.0e−7 |
| `natgas_var_targeting` — `ω` from a variance target | −5.507 | 0.080 |

Driving `ω = (1−ρ)·σ̄²` from an unconditional variance target is what made the
recursion trainable; a free `ω` drifted and wrecked QLIKE. These three predate
the current constraint code and are **not** reproducible from
`configs/natgas.yaml` — `*_ablate.yaml` variants are.

## Method notes

- **Splits.** Train / validation / test by date, set per config. Architecture
  and parameterisation were selected on validation; the test split is a single
  final look.
- **Significance.** Diebold–Mariano on per-observation QLIKE with a
  Bartlett/Newey–West HAC long-run variance (forecast losses are
  autocorrelated, so naive standard errors over-reject), plus the Model
  Confidence Set to handle multiple comparisons across the five models. Written
  to `significance.json` per run.
- **Known limitation.** DM and MCS run per-seed but are only aggregated for the
  reference seed; the seed sweep aggregates QLIKE, MSE and VaR only.

## Install

```bash
conda env create -f environment.yml && conda activate deepgarch
# or: pip install -e .
```

## Run

```bash
python run.py --config configs/spy.yaml        # also: natgas, oil, *_ablate
python sweep.py                                 # 3 markets x seeds 0-9
python sweep.py --seeds 0 1 2 --configs configs/natgas.yaml
pytest                                          # baseline tests
```

Each run writes plots, `comparison_metrics.json`, `parameter_path.csv`,
`loss_series.csv` (per-observation QLIKE/MSE for every model),
`significance.json` and a `*_model.pt` checkpoint to `output.dir`. Sweep output
lands in `results/sweep/<market>/seed_<n>/`, aggregated to
`results/sweep/summary.csv`.

## Layout

```
src/deepgarch/
  models/cond_garchnet.py   parameter constraints + variance recursion
  models/nn/paramnet.py     the MLP
  features/pipeline.py      lagged feature construction
  eval/metrics.py           QLIKE, MSE, VaR backtest, calibration
  eval/tests.py             Diebold-Mariano, Christoffersen, MCS
  eval/baselines.py         arch-fitted GARCH / GJR / EGARCH / EWMA
  data/                     yfinance loader, EIA release calendar + storage
run.py / sweep.py           single run / seed sweep
```

## License

MIT.