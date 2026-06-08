import math
from statistics import NormalDist

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Forecast-accuracy metrics
# ---------------------------------------------------------------------------

def qlike(returns, forecast_var) -> float:

    r = _to_numpy(returns)
    h = _to_numpy(forecast_var)
    if np.any(h <= 0):
        raise ValueError("forecast_var must be strictly positive.")
    return float(np.mean(np.log(h) + r ** 2 / h))


def mse_variance(returns, forecast_var) -> float:
    """
    Mean squared error between the squared-return proxy and forecast variance.

        MSE = (1/T) Σ (ε²_t − h_t)²

    A simple, scale-sensitive view that complements QLIKE. Lower is better.
    """
    r = _to_numpy(returns)
    h = _to_numpy(forecast_var)
    return float(np.mean((r ** 2 - h) ** 2))


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


# ---------------------------------------------------------------------------
# Bundling and comparison
# ---------------------------------------------------------------------------

def evaluate(returns, forecast_var, alpha: float = 0.01) -> dict:
    """
    Compute all metrics for a single model's forecasts.

    Returns
    -------
    dict with keys: qlike, mse_variance, var (the var_backtest dict).
    """
    return {
        "qlike":        qlike(returns, forecast_var),
        "mse_variance": mse_variance(returns, forecast_var),
        "var":          var_backtest(returns, forecast_var, alpha=alpha),
    }


def comparison_table(results: dict[str, dict]) -> str:

    header = f"{'model':<16} {'QLIKE':>12} {'MSE(var)':>14} {'VaR viol.':>12} {'Kupiec p':>10}"
    lines = [header, "-" * len(header)]
    for name, res in results.items():
        v = res["var"]
        lines.append(
            f"{name:<16} "
            f"{res['qlike']:>12.4f} "
            f"{res['mse_variance']:>14.3e} "
            f"{v['violation_rate']:>11.2%} "
            f"{v['kupiec_pvalue']:>10.3f}"
        )
    # Annotate the expected violation rate for reference.
    any_var = next(iter(results.values()))["var"]
    lines.append("-" * len(header))
    lines.append(f"{'(expected VaR viol. rate':<16} {'':>12} {'':>14} {any_var['expected_rate']:>11.2%} {')':>10}")
    return "\n".join(lines)