import math
from statistics import NormalDist

import numpy as np
import torch

from .tests import christoffersen, diebold_mariano


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Forecast-accuracy metrics
# ---------------------------------------------------------------------------

# Per-observation QLIKE loss L_t = log(h_t) + RV_t/h_t, scored against a
# realized-variance proxy (Parkinson range variance), not squared returns.
# Kept as a series so DM/MCS can work on the per-obs loss differential.
def qlike_series(realized_var, forecast_var) -> np.ndarray:
    rv = _to_numpy(realized_var)
    h = _to_numpy(forecast_var)
    if np.any(h <= 0):
        raise ValueError("forecast_var must be strictly positive.")
    return np.log(h) + rv / h


def qlike(realized_var, forecast_var) -> float:
    return float(np.mean(qlike_series(realized_var, forecast_var)))


def mse_variance_series(realized_var, forecast_var) -> np.ndarray:
    rv = _to_numpy(realized_var)
    h = _to_numpy(forecast_var)
    return (rv - h) ** 2


def mse_variance(realized_var, forecast_var) -> float:
    return float(np.mean(mse_variance_series(realized_var, forecast_var)))


# ---------------------------------------------------------------------------
# Risk metric: Value-at-Risk backtest
# ---------------------------------------------------------------------------

def var_backtest(returns, forecast_var, alpha: float = 0.01) -> dict:

    r = _to_numpy(returns)
    h = _to_numpy(forecast_var)
    sigma = np.sqrt(h)

    z = NormalDist().inv_cdf(alpha)           # negative quantile
    quantile = z * sigma                       # VaR threshold per timestep
    violations = r < quantile
    x = int(violations.sum())
    T = len(r)
    pi_obs = x / T

    # Kupiec POF likelihood-ratio statistic.
    # ln L0 under H0 (rate = alpha); ln L1 under observed rate.
    # 0 * log(0) is defined as 0 (the convention for the empty-count term).
    def _xlogy(a: float, b: float) -> float:
        return 0.0 if a == 0 else a * math.log(b)

    ll0 = _xlogy(x, alpha)     + _xlogy(T - x, 1 - alpha)
    ll1 = _xlogy(x, pi_obs)    + _xlogy(T - x, 1 - pi_obs)
    lr = -2.0 * (ll0 - ll1)

    # Survival function of χ²₁ via erfc — avoids a scipy dependency.
    # P(χ²₁ > lr) = erfc( sqrt(lr / 2) ).
    pvalue = math.erfc(math.sqrt(lr / 2.0)) if lr > 0 else 1.0

    return {
        "alpha":          alpha,
        "n_obs":          T,
        "n_violations":   x,
        "violation_rate": pi_obs,
        "expected_rate":  alpha,
        "kupiec_lr":      lr,
        "kupiec_pvalue":  pvalue,
    }


def residual_calibration(returns, forecast_var, alpha):
    r = _to_numpy(returns)
    h = _to_numpy(forecast_var)
    sigma = np.sqrt(h)
    z = r / sigma

    mean_z2 = np.mean(z**2)

    empirical_quantile = np.quantile(z, alpha)
    scale_corrected_quant = empirical_quantile / np.sqrt(mean_z2)
    gaussian_quantile = NormalDist().inv_cdf(alpha)

    return {
        "mean_z2": mean_z2,
        "empirical_quantile": empirical_quantile,
        "scale_corrected_quant": scale_corrected_quant,
        "gaussian_quantile": gaussian_quantile,
    }



# ---------------------------------------------------------------------------
# Bundling and comparison
# ---------------------------------------------------------------------------

# All metrics for one model's forecasts. `realized_var` (Parkinson proxy)
# scores QLIKE/MSE; `returns` drives the VaR backtest and calibration. The
# *_series values are per-obs np.ndarrays (not JSON-serializable) for DM/MCS.
def evaluate(returns, forecast_var, realized_var, alpha: float = 0.01) -> dict:
    q_series = qlike_series(realized_var, forecast_var)
    mse_series = mse_variance_series(realized_var, forecast_var)
    return {
        "qlike":               float(np.mean(q_series)),
        "qlike_series":        q_series,
        "mse_variance":        float(np.mean(mse_series)),
        "mse_variance_series": mse_series,
        "var":                 var_backtest(returns, forecast_var, alpha=alpha),
        "calibration":         residual_calibration(returns, forecast_var, alpha=alpha),
        "christoffersen":      christoffersen(returns, forecast_var, alpha=alpha),
    }


# `benchmark` keys the DM column: every other model's QLIKE loss series is
# tested against the benchmark's, so its own row shows no DM p. Christoffersen
# p is each model's own LR_cc conditional-coverage p-value.
def comparison_table(results: dict[str, dict], benchmark: str = "Static GARCH") -> str:
    header = (
        f"{'model':<16} {'QLIKE':>12} {'MSE(var)':>14} {'VaR viol.':>12} "
        f"{'Kupiec p':>10} {'z2 bias':>10} {'DM p':>10} {'Chris. p':>10}"
    )
    lines = [header, "-" * len(header)]
    bench_series = results.get(benchmark, {}).get("qlike_series")
    for name, res in results.items():
        v = res["var"]
        c = res["calibration"]
        chris_p = res["christoffersen"]["p_cc"]
        if name != benchmark and bench_series is not None:
            dm_p = diebold_mariano(res["qlike_series"], bench_series)["p_value"]
            dm_p_str = f"{dm_p:>10.3f}"
        else:
            dm_p_str = f"{'—':>10}"
        lines.append(
            f"{name:<16} "
            f"{res['qlike']:>12.4f} "
            f"{res['mse_variance']:>14.3e} "
            f"{v['violation_rate']:>11.2%} "
            f"{v['kupiec_pvalue']:>10.3f} "
            f"{c['mean_z2']:>10.3f} "
            f"{dm_p_str} "
            f"{chris_p:>10.3f}"
        )
    # Annotate the expected violation rate for reference.
    any_var = next(iter(results.values()))["var"]
    lines.append("-" * len(header))
    lines.append(
        f"{'(expected VaR viol. rate':<16} {'':>12} {'':>14} "
        f"{any_var['expected_rate']:>11.2%} {')':>10} {'':>10} {'':>10}"
    )
    return "\n".join(lines)