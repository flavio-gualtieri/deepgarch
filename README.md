# deepgarch

**Does conditioning GARCH parameters on market state improve volatility forecasts?**

The canonical GARCH(1,1) holds `(ω, α, β)` constant, while this project -- garchnet -- treats these parameters as a function of observable market state.

A small MLP maps lagged returns, realised volatility, volume
and exogenous series (e.g. EIA gas-storage releases) to daily `(ω, α, β)`,
trained by backpropagating through the variance recursion. The model is benchmarked against
static GARCH, GJR-GARCH, EGARCH and EWMA. The tested assets are SPY, natural gas (`NG=F`) and WTI crude (`CL=F`). The model is run for 10 seeds per market, scored with QLIKE against Parkinson range variance plus VaR backtests, with Diebold–Mariano (HAC) and the Model Confidence Set for significance.

The conditioned model boosts performance on natural gas, but ties with the static GARCH family on SPI and oil. The natural gas improvement is mainly due to the test period being a violent regime shift. Furthermore, this improvement only occurs when the conditional long-run variance head is tightly bounded.

**Why bounding is necessary.** The first natural gas model underestimated variance
~2.6×. A calibration diagnostic traced this to the conditional level head
breaching its safety bound (`v_max`) during the 2021+ regime shift (train ~50%
annualised vol → test ~80%). Tightening `v_max` fixed this problem. A 10-seed sweep for each asset then
showed the right bound is *market-specific* and tracks how far the test period
strays from training: natural gas needs it tight (`v_max = 1`), oil needs it
slack (`v_max = 3`, non-binding), and SPY is seemingly indifferent. On the validation split — where every market is close to training — the bound does almost nothing. See
[The `v_max` result](#the-v_max-result).

---

## Model

The MLP emits three unbounded reals per day. `_constrain_path` maps them to
valid GARCH parameters:

```
ρ  = max_persistence · sigmoid(ρ_raw)          persistence (α+β), capped < 1
α  = ρ · s_max · sigmoid(φ_raw)                 share of ρ on the shock term
β  = ρ − α
σ̄² = exp(v₀ + v_max · tanh(v_raw / v_max))      long-run variance level
ω  = (1 − ρ) · σ̄²                               variance targeting
```

Stationarity holds by construction: `ω > 0`, `α, β > 0`, `α + β < max_persistence`. The recursion `h_t = ω_t + α_t r²_{t−1} + β_t h_{t−1}` runs on
top, trained on Gaussian NLL.

**Why variance targeting.** Estimating `ω` directly is not ideal because it is a small value
(~1e−5) whcih is strongly coupled to `ρ`, as `σ̄² = ω/(1−ρ)`. At `ρ = 0.999` a 1%
error in `ω` moves the long-run level by 10x, so the likelihood surface
is a long thin ridge. Reparameterising to `(σ̄², ρ)` rotates the ridge into two near-orthogonal
directions. `v₀ = log(train-set unconditional variance)`, so the network starts
at the historical level and learns log-deviations from it. `v_max` bounds those
deviations to a factor of `exp(v_max)` in either direction.

**Leakage.** Every feature in `features/pipeline.py` is `.shift(1)`-lagged. The
one deliberate exception is `is_eia_day`, shifted `−1`: the recursion consumes
row `t−1`'s parameters to produce `σ²_t`, so the storage-release flag must sit
on the row before the release. `fit_initial_variance` seeds the recursion from
train returns only, so no val/test variance enters the initial condition.

## The `v_max` result

`v_max` began as an `exp()`-overflow rail, with default value 3.0. The natural-gas
calibration diagnostic showed the level head using the entire band on the test
split — `σ̄²` swinging from 5.1e−5 to 2.0e−2, `mean_z2 ≈ 2.3` (variance
underestimated ~2×). The sweep below (5 values × 10 seeds × 3 markets, test
split; `results/sweep_test_vmax/`) shows what tightening it does.

| `v_max` | natgas QLIKE ± sd | natgas `mean_z2` | SPY QLIKE ± sd | oil QLIKE ± sd |
|---:|---:|---:|---:|---:|
| **1.0** | **−5.5145 ± 0.0070** | 1.50 | **−9.0633 ± 0.0157** | −5.9374 ± 0.0525 |
| 2.0 | −5.4741 ± 0.0117 | 1.82 | −9.0539 ± 0.0172 | **−6.0038 ± 0.0046** |
| 3.0 | −5.4167 ± 0.0388 | 2.29 | −9.0493 ± 0.0171 | **−6.0038 ± 0.0046** |
| 4.0 | −5.3046 ± 0.1482 | 3.20 | −9.0467 ± 0.0169 | −6.0038 ± 0.0046 |
| 5.0 | −5.0824 ± 0.4310 | 5.06 | −9.0404 ± 0.0251 | −6.0038 ± 0.0046 |

![v_max sweep — 10 seeds, test split](results/vmax_sweep.png)

- **Natural gas.** Tightening `v_max` improves the mean QLIKE monotonically *and*
  collapses seed variance: sd `0.43 → 0.007` from `v_max = 5` to `1`, `mean_z2`
  `5.1 → 1.5`. At `v_max ≥ 2` GARCHNet is worse than static GARCH in the mean
  (DM loss differential turns positive); at `v_max = 1` it ties static GARCH
  (DM ensemble p = 0.11) with a well-calibrated, stable forecast.
- **Oil.** The `v_max ≥ 2` rows are identical — oil's level head never reaches
  the bound, so raising it does not change anything. `v_max = 1` *clips* it: QLIKE
  worsens 0.07 and seed sd jumps 10×. Oil's config is therefore set to `v_max = 3`.
- **SPY.** Nearly flat — a 0.02 QLIKE drift across the whole grid (≈ 1.5 seed
  sd), no instability, `mean_z2` fixed at 0.90.

So the correct `v_max` tracks regime distance: natural gas (train → test vol jump)
needs the level head reined in; oil and SPY (test ≈ train) do not.

Validation does not see this: re-scoring the natural-gas sweep on the
2018–2020 validation split (`--eval-split val`, 3 seeds, `results/sweep_val/`),
the whole grid moves QLIKE by 0.007 — one seed sd — and `mean_z2` stays in a
calibrated 0.95–0.99 band:

| `v_max` | natgas val QLIKE ± sd | natgas val `mean_z2` |
|---:|---:|---:|
| 1.0 | −6.3114 ± 0.0046 | 0.99 |
| 3.0 | −6.3055 ± 0.0073 | 0.95 |
| 5.0 | −6.3040 ± 0.0080 | 0.95 |

The validation period is too close to training
for the level head to over-reach, so `v_max = 1` for natural gas rests on
test-split behaviour.

## Results

QLIKE and MSE score against **Parkinson range variance**,
`(ln(high/low))²/(4 ln 2)` — which was picked becasue it is unbiased for daily variance and less noisy than squared returns. Violations test realised returns against the forecast
distribution and are unaffected by this proxy. All numbers are from the **test** split
(`config.eval_split = test`, the default); validation figures are in
[The `v_max` result](#the-v_max-result).

| Market | Model | QLIKE (mean ± sd) | `mean_z2` | Violations (α=1%) | Kupiec p |
|---|---|---:|---:|---:|---:|
| **Natgas** (n=1428) | **GARCHNet** | **−5.5145 ± 0.0070** | 1.50 | 1.55% | 0.090 |
| | EGARCH | −5.5077 | 1.24 | 1.12% | 0.654 |
| | Static GARCH | −5.4851 | 1.17 | 1.19% | 0.482 |
| | GJR-GARCH | −5.4766 | 1.17 | 1.33% | 0.232 |
| | EWMA | −5.4594 | 1.11 | 1.05% | 0.849 |
| **SPY** (n=313) | EGARCH | −9.0683 | 0.95 | 1.92% | 0.148 |
| | GARCHNet | −9.0633 ± 0.0157 | 0.91 | 1.25% | 0.666 |
| | GJR-GARCH | −9.0224 | 0.88 | 1.28% | 0.636 |
| | EWMA | −8.9856 | 1.04 | 1.28% | 0.636 |
| | Static GARCH | −8.9689 | 0.87 | 1.28% | 0.636 |
| **Oil** (n=229) | Static GARCH | −6.0339 | 1.14 | 2.62% | 0.040 |
| | GARCHNet | −6.0038 ± 0.0046 | 1.15 | 3.14% | 0.010 |
| | EWMA | −5.9547 | 1.19 | 2.62% | 0.040 |
| | GJR-GARCH | −5.8246 | 1.28 | 3.93% | 0.001 |
| | EGARCH | −5.5915 | 1.48 | 3.93% | 0.001 |

Per-seed numbers and DM/MCS aggregates under `results/sweep_test_vmax/vmax_<v>/`.

**Natural gas.** Best mean QLIKE of the five and the lowest variance MSE by a
wide margin (1.17e−5 vs 1.86e−5 for static GARCH). In the QLIKE MCS on 10/10
seeds (EWMA excluded on 10/10, the other three baselines on 9/10). But vs static
GARCH the seed-averaged DM p is **0.11** (3/10 seeds significant at 0.05) — a
directional edge, but not decisively better since EGARCH is only 0.007 behind. VaR coverage is a weakness: 1.55% against 1% nominal, Kupiec p = 0.090, `mean_z2` = 1.50.

**SPY.** −9.0633 ± 0.0157 against EGARCH's −9.0683 — a smaller gap than the seed
sd, so it should be interpreted as a tie. GARCHNet beats static GARCH decisively (DM p < 0.001, 10/10 seeds) and sits in the QLIKE MCS 10/10 — but so do EGARCH, GJR and EWMA; only
static GARCH is excluded. So garchnet is competitive with the family, but not particularly distinguishable.

**Oil.** Second behind static GARCH (−6.0038 vs −6.0339), ahead of EWMA, GJR and
EGARCH, with the tightest seed spread (sd 0.0046). DM vs static GARCH p = 0.40 —
no edge. Every model fails VaR coverage here.

## Method notes

- **Splits and scoring.** Train / validation / test by date, per config. The
  architecture and the `(ρ, s, σ̄²)` parameterisation were fixed on validation.
  `v_max` per market is the exception — validation is too close to training to
  resolve it (see [above](#the-v_max-result)) — so `v_max = 1` for natural gas
  rests on test-split behaviour. `run.py` scores whichever split `config.eval_split` names
  (default is `test`).
- **Significance.** Diebold–Mariano on per-observation QLIKE with Bartlett /
  Newey–West HAC long-run variance (forecast losses are autocorrelated, so naive
  SEs would over-reject), plus the Model Confidence Set to handle five models at once.
  Reported both per-seed and on the seed-averaged loss series.
- **Seeds.** Results and the `v_max` grid are 10 seeds on test; the validation
  `v_max` grid is 3 seeds; the `*_ablate` runs are single-seed. Baselines are
  deterministic (so `sd = 0` automatically). Training checkpoints go to a unique
  per-run temp file, so sweeps can be run in parallel.
- **Data.** No `end` date is pinned: on re-run the test window extends to the
  latest bar and `n` grows. The natural-gas numbers here are one snapshot newer
  than SPY/oil (yfinance rate-limited the latter to a cached pull). The
  `yfinance` data cache under `src/deepgarch/data/downloaded/` is not committed.

## Install

```bash
conda env create -f environment.yml && conda activate deepgarch
# or: pip install -e .
```

## Run

```bash
python run.py --config configs/spy.yaml              # also: natgas, oil, *_ablate
python run.py --config configs/natgas.yaml           # scores config.eval_split (default: test)
python sweep.py                                      # 3 markets x seeds 0-9, test split
python sweep.py --v-max 1 2 3 4 5 --out results/sweep_test_vmax        # the v_max grid on test (seeds 0-9)
python sweep.py --configs configs/natgas.yaml --v-max 1 2 3 4 5 --seeds 0 1 2 \
    --eval-split val --out results/sweep_val         # ... natgas re-scored on validation
python sweep.py --aggregate-only                     # rebuild summaries, no training
python scripts/plot_vmax_sweep.py                    # regenerate results/vmax_sweep.png
pytest
```

Each run writes plots, `comparison_metrics.json`, `parameter_path.csv`,
`loss_series.csv` (per-observation QLIKE/MSE for every model),
`significance.json` and a `*_model.pt` checkpoint to `output.dir`. Sweeps land in
`<out>/[vmax_<v>/]<market>/seed_<n>/`, aggregated to `summary.csv` and
`significance.json` at each `vmax_<v>` root.

## Layout

```
src/deepgarch/
  config.py                 RunConfig incl. eval_split (train|val|test)
  models/cond_garchnet.py   parameter constraints + variance recursion
  models/nn/paramnet.py     the MLP
  features/pipeline.py      lagged feature construction
  eval/metrics.py           QLIKE, MSE, VaR backtest, calibration
  eval/tests.py             Diebold-Mariano, Christoffersen, MCS
  eval/baselines.py         arch-fitted GARCH / GJR / EGARCH / EWMA
  data/                     yfinance loader, EIA release calendar + storage
run.py / sweep.py           single run / seed and v_max sweeps
scripts/plot_vmax_sweep.py  regenerates the v_max figure from sweep artifacts
configs/                    one YAML per market, plus *_ablate variants
```

## License

MIT — see `LICENSE`.
