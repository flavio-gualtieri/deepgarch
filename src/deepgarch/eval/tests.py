import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from arch.bootstrap import MCS
from arch.covariance.kernel import Bartlett


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _xlogy(a: float, b: float) -> float:
    return 0.0 if a == 0 else a * math.log(b)


def _var_violations(returns, forecast_var, alpha: float) -> np.ndarray:
    r = _to_numpy(returns)
    h = _to_numpy(forecast_var)
    sigma = np.sqrt(h)
    z = NormalDist().inv_cdf(alpha)          # negative quantile
    return r < z * sigma


# ---------------------------------------------------------------------------
# Diebold-Mariano: pairwise equal-predictive-accuracy test on loss series
# ---------------------------------------------------------------------------

def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """
    Diebold-Mariano test of equal predictive accuracy between two aligned
    per-observation loss series (e.g. qlike_series for two models).

    The long-run variance of the loss differential d_t = loss_a_t - loss_b_t
    is estimated with a Bartlett/Newey-West HAC kernel, bandwidth
    floor(4*(T/100)^(2/9)) (Newey & West 1994's plug-in rule), floored at
    h-1 so an h-step forecast horizon's known MA(h-1) overlap is always
    covered. p-value is the two-sided normal-approximation p-value (as in
    the original Diebold-Mariano 1995 test).

    Returns
    -------
    dict with keys: dm_stat, p_value, mean_loss_diff (a - b; negative means
    a has lower average loss), bandwidth, h, n_obs.
    """
    a = _to_numpy(loss_a)
    b = _to_numpy(loss_b)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape.")

    d = a - b
    T = len(d)
    bandwidth = max(int(np.floor(4 * (T / 100) ** (2 / 9))), h - 1)

    dbar = float(np.mean(d))
    long_run_var = float(Bartlett(d, bandwidth=bandwidth, force_int=True).cov.long_run[0, 0])
    se = math.sqrt(long_run_var / T) if long_run_var > 0 else 0.0

    if se > 0:
        dm_stat = dbar / se
        p_value = math.erfc(abs(dm_stat) / math.sqrt(2))
    else:
        dm_stat = 0.0
        p_value = 1.0

    return {
        "dm_stat":        dm_stat,
        "p_value":        p_value,
        "mean_loss_diff": dbar,
        "bandwidth":      bandwidth,
        "h":              h,
        "n_obs":          T,
    }


# ---------------------------------------------------------------------------
# Christoffersen (1998) conditional-coverage test
# ---------------------------------------------------------------------------

def christoffersen(returns, forecast_var, alpha: float = 0.01) -> dict:
    """
    Decomposes the VaR backtest into unconditional coverage (LR_uc, the
    Kupiec POF statistic) and independence (LR_ind, whether violations
    cluster instead of scattering like a Bernoulli(alpha) process), via a
    first-order Markov chain on the violation indicator. LR_cc = LR_uc +
    LR_ind ~ chi2(2) is the joint conditional-coverage test.

    Returns
    -------
    dict with keys: alpha, n_obs, n_violations, lr_uc, lr_ind, lr_cc,
    p_ind, p_cc.
    """
    violations = _var_violations(returns, forecast_var, alpha)
    T = len(violations)
    x = int(violations.sum())

    pi_obs = x / T
    ll0 = _xlogy(x, alpha) + _xlogy(T - x, 1 - alpha)
    ll1 = _xlogy(x, pi_obs) + _xlogy(T - x, 1 - pi_obs)
    lr_uc = -2.0 * (ll0 - ll1)

    prev, curr = violations[:-1], violations[1:]
    n00 = int(np.sum(~prev & ~curr))
    n01 = int(np.sum(~prev & curr))
    n10 = int(np.sum(prev & ~curr))
    n11 = int(np.sum(prev & curr))

    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    pi01 = _safe_div(n01, n00 + n01)   # P(violation | no violation yesterday)
    pi11 = _safe_div(n11, n10 + n11)   # P(violation | violation yesterday)
    pi   = _safe_div(n01 + n11, n00 + n01 + n10 + n11)

    ll_indep = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
                + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    ll_pooled = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    lr_ind = 2.0 * (ll_indep - ll_pooled)

    lr_cc = lr_uc + lr_ind

    p_ind = math.erfc(math.sqrt(lr_ind / 2.0)) if lr_ind > 0 else 1.0
    p_cc = math.exp(-lr_cc / 2.0)   # chi2(2) survival function has a closed form

    return {
        "alpha":        alpha,
        "n_obs":        T,
        "n_violations": x,
        "lr_uc":        lr_uc,
        "lr_ind":       lr_ind,
        "lr_cc":        lr_cc,
        "p_ind":        p_ind,
        "p_cc":         p_cc,
    }


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# ---------------------------------------------------------------------------

def model_confidence_set(
    losses: dict[str, np.ndarray],
    alpha: float = 0.10,
    reps: int = 5000,
    block_size: int | None = None,
    bootstrap: str = "stationary",
    seed: int | None = 0,
) -> dict:
    """
    Model Confidence Set over named per-observation loss series (e.g. each
    model's qlike_series), via arch.bootstrap.MCS — already a project
    dependency (used for the static GARCH baseline) and a tested
    implementation of the bootstrap elimination procedure, so it's reused
    here rather than re-derived.

    Returns
    -------
    dict with keys: alpha, included (names in the MCS), excluded (names
    eliminated), pvalues (name -> MCS p-value).
    """
    names = list(losses.keys())
    frame = pd.DataFrame({name: _to_numpy(losses[name]) for name in names})

    mcs = MCS(frame, size=alpha, reps=reps, block_size=block_size, bootstrap=bootstrap, seed=seed)
    mcs.compute()

    return {
        "alpha":    alpha,
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "pvalues":  mcs.pvalues["Pvalue"].to_dict(),
    }
