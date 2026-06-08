from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# Known volatility events to annotate. Extend as needed.
_DEFAULT_EVENTS = {
    "GFC 2008":   "2008-09-15",   # Lehman collapse
    "COVID 2020": "2020-03-16",   # COVID crash
    "Rate hikes": "2022-06-13",   # 2022 selloff
}


def _resolve_x(index, n: int):
    """Return an x-axis array: the given date index, or a 0..n range."""
    if index is not None:
        return pd.DatetimeIndex(index)
    return np.arange(n)


def _annotate_events(ax, index, events: dict[str, str]) -> None:
    """Draw vertical lines at event dates that fall within the index range."""
    if index is None:
        return
    idx = pd.DatetimeIndex(index)
    lo, hi = idx[0], idx[-1]
    for label, date in events.items():
        ts = pd.Timestamp(date)
        if lo <= ts <= hi:
            ax.axvline(ts, color="firebrick", linestyle=":", linewidth=1, alpha=0.7)
            ax.text(
                ts, ax.get_ylim()[1], f" {label}",
                rotation=90, va="top", ha="left",
                fontsize=7, color="firebrick", alpha=0.8,
            )


def plot_parameter_paths(
    omega, alpha, beta,
    index=None,
    events: dict[str, str] | None = None,
    save_path: str | None = None,
):
    """
    Plot the time-varying GARCH parameters produced by the network.

    Parameters
    ----------
    omega : array-like (T,)
    alpha : array-like (T,) or (T, 1)
    beta  : array-like (T,) or (T, 1)
    index : optional date index of length T for the x-axis.
    events : optional {label: date} markers. Defaults to GFC/COVID/2022.
    save_path : if given, save the figure there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    o = _to_numpy(omega).reshape(-1)
    a = _to_numpy(alpha).reshape(-1)
    b = _to_numpy(beta).reshape(-1)
    persistence = a + b
    events = _DEFAULT_EVENTS if events is None else events
    x = _resolve_x(index, len(o))

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    panels = [
        (axes[0], o,           r"$\omega_t$ (baseline variance)", "tab:blue"),
        (axes[1], a,           r"$\alpha_t$ (ARCH / shock term)",  "tab:orange"),
        (axes[2], b,           r"$\beta_t$ (GARCH / memory term)", "tab:green"),
        (axes[3], persistence, r"$\alpha_t + \beta_t$ (persistence)", "tab:red"),
    ]
    for ax, series, title, color in panels:
        ax.plot(x, series, color=color, linewidth=1.0)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.2)
        _annotate_events(ax, index, events)

    axes[3].axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[3].set_xlabel("Date" if index is not None else "Time step")
    fig.suptitle("NeuralGARCH: time-varying parameter paths", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_volatility_comparison(
    returns,
    neural_var,
    static_var=None,
    index=None,
    events: dict[str, str] | None = None,
    save_path: str | None = None,
):
    """
    Compare conditional volatility (σ_t = sqrt(h_t)) across models.

    Plots |returns| in the background with the NeuralGARCH volatility and,
    optionally, the static GARCH volatility on top.

    Parameters
    ----------
    returns : array-like (T,)
    neural_var : array-like (T,)   conditional variance from NeuralGARCH
    static_var : optional array-like (T,)  conditional variance from baseline
    index, events, save_path : as in plot_parameter_paths.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    r = _to_numpy(returns).reshape(-1)
    events = _DEFAULT_EVENTS if events is None else events
    x = _resolve_x(index, len(r))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, np.abs(r), color="lightgrey", linewidth=0.6, label="|returns|")
    ax.plot(x, np.sqrt(_to_numpy(neural_var).reshape(-1)),
            color="tab:blue", linewidth=1.1, label="NeuralGARCH σ")
    if static_var is not None:
        ax.plot(x, np.sqrt(_to_numpy(static_var).reshape(-1)),
                color="tab:orange", linewidth=1.1, alpha=0.85, label="Static GARCH σ")

    ax.set_ylabel("Conditional volatility")
    ax.set_xlabel("Date" if index is not None else "Time step")
    ax.set_title("Conditional volatility: adaptive vs. static")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    _annotate_events(ax, index, events)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_var_violations(
    returns,
    forecast_var,
    alpha: float = 0.01,
    index=None,
    save_path: str | None = None,
):
    """
    Plot returns with the model's VaR band and mark violations.

    Parameters
    ----------
    returns : array-like (T,)
    forecast_var : array-like (T,)
    alpha : VaR tail probability.
    index, save_path : as above.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from statistics import NormalDist

    r = _to_numpy(returns).reshape(-1)
    sigma = np.sqrt(_to_numpy(forecast_var).reshape(-1))
    z = NormalDist().inv_cdf(alpha)
    var_line = z * sigma                  # lower VaR threshold (negative)
    violations = r < var_line
    x = _resolve_x(index, len(r))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, r, color="lightsteelblue", linewidth=0.6, label="returns")
    ax.plot(x, var_line, color="tab:red", linewidth=1.0,
            label=f"{1 - alpha:.0%} VaR")
    ax.scatter(
        np.asarray(x)[violations], r[violations],
        color="black", s=12, zorder=5,
        label=f"violations ({int(violations.sum())})",
    )
    ax.set_ylabel("Return")
    ax.set_xlabel("Date" if index is not None else "Time step")
    ax.set_title(f"VaR backtest at {1 - alpha:.0%} confidence")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig