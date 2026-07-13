

import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


torch.manual_seed(42)
np.random.seed(42)

PLOTS_DIR = "plots_oil"
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def to_return_tensor(frame: pd.DataFrame) -> Tensor:
    return torch.tensor(frame["returns"].to_numpy(dtype=np.float32), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 2026 Iran war / Strait of Hormuz crisis — event timeline used to annotate plots.
# Sourced from Wikipedia ("Economic impact of the 2026 Iran war",
# "2026 Strait of Hormuz crisis") and CNBC / Al Jazeera reporting, July 2026.
# ---------------------------------------------------------------------------

EVENTS = [
    ("2026-02-28", "US-Israel strikes begin"),
    ("2026-03-04", "Hormuz closure announced"),
    ("2026-04-06", "WTI peak (~$112)"),
    ("2026-06-17", "Islamabad Memorandum"),
    ("2026-06-20", "Iran re-closes Hormuz"),
    ("2026-07-12", "US strikes air defenses"),
]


def annotate_events(ax, y_frac: float = 0.92) -> None:
    ymin, ymax = ax.get_ylim()
    for date_str, label in EVENTS:
        d = pd.Timestamp(date_str)
        if d < mdates.num2date(ax.get_xlim()[0]).replace(tzinfo=None) or d > mdates.num2date(ax.get_xlim()[1]).replace(tzinfo=None):
            continue
        ax.axvline(d, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
        ax.text(d, ymin + y_frac * (ymax - ymin), " " + label, rotation=90,
                 fontsize=6.5, va="top", ha="center", alpha=0.75)


# ---------------------------------------------------------------------------
# Conditional GARCHNet — feature-conditioned GARCH(1,1).
# (Same architecture used in run_natgas.py; duplicated here per the repo's
# existing pattern of keeping each commodity script self-contained.)
# ---------------------------------------------------------------------------

class ConditionalGARCHNet(nn.Module):

    _VALID_CONSTRAINTS = ("none", "positive", "stationary")

    def __init__(
        self,
        paramnet: nn.Module,
        p: int = 1,
        q: int = 1,
        constraint: str = "stationary",
        max_persistence: float = 0.995,
    ) -> None:
        super().__init__()

        if p != 1 or q != 1:
            raise NotImplementedError("Start with conditional GARCH(1,1).")
        if constraint not in self._VALID_CONSTRAINTS:
            raise ValueError(f"constraint must be one of {self._VALID_CONSTRAINTS}.")

        self.paramnet = paramnet
        self.p = p
        self.q = q
        self.constraint = constraint
        self.max_persistence = max_persistence

        expected = 1 + q + p
        n_params = getattr(paramnet, "_n_params", None)
        if n_params != expected:
            raise ValueError(
                f"paramnet must output {expected} params for GARCH({p},{q}); "
                f"got {n_params}."
            )

    def _split(self, raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected = 1 + self.q + self.p
        if raw.ndim != 2 or raw.shape[1] != expected:
            raise ValueError(
                f"Expected raw shape (T, {expected}); got {tuple(raw.shape)}."
            )

        omega_raw = raw[:, 0]
        alpha_raw = raw[:, 1 : 1 + self.q]
        beta_raw = raw[:, 1 + self.q : 1 + self.q + self.p]
        return omega_raw, alpha_raw, beta_raw

    def _constrain_path(
        self,
        omega_raw: Tensor,
        alpha_raw: Tensor,
        beta_raw: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        eps = 1e-8

        if self.constraint == "none":
            return omega_raw, alpha_raw, beta_raw

        omega = F.softplus(omega_raw) + eps

        if self.constraint == "positive":
            return omega, F.softplus(alpha_raw), F.softplus(beta_raw)

        slack = alpha_raw.new_zeros(alpha_raw.shape[0], 1)
        logits = torch.cat([alpha_raw, beta_raw, slack], dim=-1)
        weights = torch.softmax(logits, dim=-1) * self.max_persistence

        alpha = weights[:, : self.q]
        beta = weights[:, self.q : self.q + self.p]
        return omega, alpha, beta

    def parameter_path(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.paramnet(X)
        omega_raw, alpha_raw, beta_raw = self._split(raw)
        return self._constrain_path(omega_raw, alpha_raw, beta_raw)

    def variance_path(
        self,
        returns: Tensor,
        omega: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Tensor:
        if returns.ndim != 1:
            raise ValueError(f"returns must be 1D; got {tuple(returns.shape)}.")

        T = returns.shape[0]
        if T < 2:
            raise ValueError("Need at least two returns for a GARCH recursion.")

        initial_var = returns.var(unbiased=False).clamp_min(1e-8)
        sigma2_values = [initial_var]

        for t in range(1, T):
            prev_sigma2 = sigma2_values[-1]
            sigma2_t = (
                omega[t - 1]
                + alpha[t - 1, 0] * returns[t - 1].pow(2)
                + beta[t - 1, 0] * prev_sigma2
            )
            sigma2_values.append(sigma2_t.clamp_min(1e-8))

        return torch.stack(sigma2_values)

    @staticmethod
    def negative_loglikelihood(returns: Tensor, sigma2: Tensor) -> Tensor:
        return 0.5 * torch.mean(torch.log(sigma2) + returns.pow(2) / sigma2)

    def diagnostics(self, X: Tensor, returns: Tensor) -> dict[str, Tensor]:
        omega, alpha, beta = self.parameter_path(X)
        sigma2 = self.variance_path(returns, omega, alpha, beta)
        alpha_1 = alpha[:, 0]
        beta_1 = beta[:, 0]
        return {
            "omega": omega,
            "alpha": alpha_1,
            "beta": beta_1,
            "persistence": alpha_1 + beta_1,
            "sigma2": sigma2,
            "sigma": torch.sqrt(sigma2),
        }

    def forecast_fixed_params(
        self,
        X: Tensor,
        returns: Tensor,
        h: int = 30,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if h <= 0:
            raise ValueError("h must be positive.")

        diag = self.diagnostics(X, returns)

        omega_T = diag["omega"][-1]
        alpha_T = diag["alpha"][-1]
        beta_T = diag["beta"][-1]
        persistence_T = alpha_T + beta_T

        sigma2_next = (
            omega_T
            + alpha_T * returns[-1].pow(2)
            + beta_T * diag["sigma2"][-1]
        ).clamp_min(1e-8)

        forecasts = [sigma2_next]
        for _ in range(1, h):
            sigma2_next = (omega_T + persistence_T * sigma2_next).clamp_min(1e-8)
            forecasts.append(sigma2_next)

        params_T = {
            "omega": omega_T,
            "alpha": alpha_T,
            "beta": beta_T,
            "persistence": persistence_T,
            "last_sigma2": diag["sigma2"][-1],
        }
        return torch.stack(forecasts), params_T

    def forward(self, X: Tensor, returns: Tensor) -> Tensor:
        if X.shape[0] != returns.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} timesteps but returns has {returns.shape[0]}."
            )
        omega, alpha, beta = self.parameter_path(X)
        sigma2 = self.variance_path(returns, omega, alpha, beta)
        return self.negative_loglikelihood(returns, sigma2)


# ---------------------------------------------------------------------------
# Data — WTI crude (CL=F) as primary series, refined-product / gas complex as
# cross-market auxiliary features.
# ---------------------------------------------------------------------------

from deepgarch.data.natural_gas_loader import NaturalGasMarketData

data = NaturalGasMarketData(
    ticker="CL=F",
    start="2005-01-01",
    end="2026-07-13",
    val_start="2023-01-01",
    test_start="2025-10-01",
    yahoo_aux_tickers={
        "natural_gas": "NG=F",
        "heating_oil": "HO=F",
        "gasoline": "RB=F",
    },
).load()

data.summary()

train_frame = data.train.loc[data.train["returns"].notna()].copy()
val_frame = data.val.loc[data.val["returns"].notna()].copy()
test_frame = data.test.loc[data.test["returns"].notna()].copy()
all_frame = pd.concat([train_frame, val_frame, test_frame], axis=0)

n_train = len(train_frame)
n_val = len(val_frame)
n_test = len(test_frame)
test_start_idx = n_train + n_val


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

from deepgarch.features.natural_gas_features import default_natural_gas_feature_pipeline

pipeline = default_natural_gas_feature_pipeline()
pipeline.fit(train_frame)
X_all = pipeline.transform(all_frame)

X_train = X_all[:n_train]
X_val = X_all[n_train : n_train + n_val]
X_test = X_all[test_start_idx:]

returns_all = to_return_tensor(all_frame)
returns_train = returns_all[:n_train]
returns_val = returns_all[n_train : n_train + n_val]
returns_test = returns_all[test_start_idx:]

print(f"\n[features] n_features={pipeline.n_features}")
print("first 15 features:", pipeline.feature_names[:15])
print(f"splits: train={len(returns_train)}, val={len(returns_val)}, test={len(returns_test)}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

from deepgarch.models.nn import ParamNet

p, q = 1, 1

paramnet = ParamNet(
    embedding_dim=pipeline.n_features,
    hidden_dims=[64, 32],
    n_params=1 + q + p,
    dropout=0.10,
)

model = ConditionalGARCHNet(
    paramnet=paramnet,
    p=p,
    q=q,
    constraint="stationary",
    max_persistence=0.995,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

from deepgarch.train import Trainer, TrainConfig, TrainingResult

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class TqdmTrainer(Trainer):
    def fit(self, X_train, returns_train, X_val, returns_val):
        if tqdm is None:
            return super().fit(X_train, returns_train, X_val, returns_val)

        result = TrainingResult()
        checkpoint = Path(self.config.checkpoint_path)
        t0 = time.perf_counter()
        epochs_without_improvement = 0

        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"

        with tqdm(
            total=self.config.max_epochs,
            desc="training",
            unit="ep",
            bar_format=bar_fmt,
            dynamic_ncols=True,
        ) as pbar:
            for epoch in range(self.config.max_epochs):
                train_loss = self._train_step(X_train, returns_train)
                val_loss = self._eval_step(X_val, returns_val)
                self.scheduler.step(val_loss)

                result.train_losses.append(train_loss)
                result.val_losses.append(val_loss)

                improvement = result.best_val_loss - val_loss
                if improvement > self.config.min_delta:
                    result.best_val_loss = val_loss
                    result.best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), checkpoint)
                else:
                    epochs_without_improvement += 1

                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    train=f"{train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    lr=f"{lr:.2e}",
                    best=f"{result.best_val_loss:.4f}",
                    refresh=False,
                )
                pbar.update(1)

                if epochs_without_improvement >= self.config.patience:
                    pbar.write(
                        f"  early stop at epoch {epoch + 1} "
                        f"(best val={result.best_val_loss:.4f})"
                    )
                    result.stopped_early = True
                    break

        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
            checkpoint.unlink()

        result.elapsed_seconds = time.perf_counter() - t0
        return result


config = TrainConfig(
    max_epochs=500,
    learning_rate=2e-3,
    weight_decay=1e-4,
    patience=100,
    min_delta=1e-4,
    grad_clip=1.0,
    log_every=25,
)

trainer = TqdmTrainer(model, config)
result = trainer.fit(X_train, returns_train, X_val, returns_val)


# ---------------------------------------------------------------------------
# Plots: training curve
# ---------------------------------------------------------------------------

print("\n[plots]")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.train_losses, label="train NLL", linewidth=1.2)
ax.plot(result.val_losses, label="val NLL", linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("NLL")
ax.set_title("WTI Crude Conditional GARCHNet training curve")
ax.legend()
fig.tight_layout()
savefig("01_training_curve.png")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def fallback_evaluate(returns_np: np.ndarray, variance_np: np.ndarray) -> dict[str, float]:
    variance_np = np.maximum(variance_np, 1e-12)
    nll = 0.5 * np.mean(np.log(variance_np) + returns_np ** 2 / variance_np)
    mse_r2 = np.mean((returns_np ** 2 - variance_np) ** 2)
    qlike = np.mean(np.log(variance_np) + returns_np ** 2 / variance_np)
    return {"nll": float(nll), "qlike": float(qlike), "mse_r2": float(mse_r2)}

try:
    from deepgarch.eval import evaluate, comparison_table, StaticGARCH
except ImportError:
    evaluate = fallback_evaluate
    comparison_table = None
    StaticGARCH = None

model.eval()
with torch.no_grad():
    diag_all = model.diagnostics(X_all, returns_all)

neural_var_all = diag_all["sigma2"].detach().cpu().numpy()
neural_var = neural_var_all[test_start_idx:]

rets_test_np = returns_test.detach().cpu().numpy()
neural_metrics = evaluate(rets_test_np, neural_var)

static_var = None
static_metrics = None
if StaticGARCH is not None:
    static_garch = StaticGARCH()
    static_garch.fit(train_frame["returns"])

    static_var_all = np.asarray(static_garch.filter(all_frame["returns"]))
    static_var = static_var_all[test_start_idx:]
    static_metrics = evaluate(rets_test_np, static_var)

if comparison_table is not None and static_metrics is not None:
    comparison_table({
        "Conditional Oil-GARCHNet": neural_metrics,
        "Static GARCH": static_metrics,
    })
else:
    print("\n[metrics]")
    print("Conditional Oil-GARCHNet:", neural_metrics)
    if static_metrics is not None:
        print("Static GARCH:", static_metrics)


params = pd.DataFrame(
    {
        "omega": diag_all["omega"].detach().cpu().numpy(),
        "alpha": diag_all["alpha"].detach().cpu().numpy(),
        "beta": diag_all["beta"].detach().cpu().numpy(),
        "persistence": diag_all["persistence"].detach().cpu().numpy(),
        "sigma2": diag_all["sigma2"].detach().cpu().numpy(),
        "sigma": diag_all["sigma"].detach().cpu().numpy(),
    },
    index=all_frame.index,
)
params_path = os.path.join(PLOTS_DIR, "conditional_parameter_path.csv")
params.to_csv(params_path)
print(f"  saved -> {params_path}")


# ---------------------------------------------------------------------------
# Plot: forecasted vol vs realised |return|, annotated with the war timeline
# ---------------------------------------------------------------------------

dates = test_frame.index
realised = np.abs(rets_test_np)
neural_vol = np.sqrt(neural_var)
static_vol = np.sqrt(static_var) if static_var is not None else None

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].fill_between(dates, realised, alpha=0.25, label="|return| realised proxy")
axes[0].plot(dates, neural_vol, linewidth=1.3, label="Conditional Oil-GARCHNet")
if static_vol is not None:
    axes[0].plot(dates, static_vol, linewidth=1.0, linestyle="--", label="Static GARCH(1,1)")
axes[0].set_ylabel("Daily volatility")
axes[0].set_title("WTI crude: forecasted sigma vs realised |return| through the 2026 Iran war")
axes[0].legend(loc="upper left")
annotate_events(axes[0])

if static_vol is not None:
    axes[1].plot(dates, neural_vol - static_vol, linewidth=0.8)
    axes[1].axhline(0, linewidth=0.7, linestyle="--")
    axes[1].set_ylabel("Conditional − Static")
    axes[1].set_title("Difference: conditional model vs static baseline")
else:
    axes[1].plot(dates, neural_vol, linewidth=0.8)
    axes[1].set_ylabel("Conditional sigma")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
savefig("02_forecast_vs_realised.png")


# ---------------------------------------------------------------------------
# Plot: parameter path on test period (annotated)
# ---------------------------------------------------------------------------

params_test = params.iloc[test_start_idx:]
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
axes[0].plot(params_test.index, params_test["alpha"], linewidth=1.0, label="alpha_t")
axes[0].plot(params_test.index, params_test["beta"], linewidth=1.0, label="beta_t")
axes[0].set_title("WTI crude — conditional GARCH parameters through the 2026 Iran war")
axes[0].legend(loc="upper left")
axes[1].plot(params_test.index, params_test["persistence"], linewidth=1.2, color="darkred")
axes[1].set_ylabel("alpha + beta (persistence)")
annotate_events(axes[1])
axes[2].plot(params_test.index, params_test["omega"], linewidth=1.0)
axes[2].set_ylabel("omega")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[2].tick_params(axis="x", rotation=30)
fig.tight_layout()
savefig("03_parameter_path.png")


# ---------------------------------------------------------------------------
# Plot: VaR violations
# ---------------------------------------------------------------------------

z99 = 2.326
var_models = [("Conditional Oil-GARCHNet", neural_vol, "04_var_violations_conditional.png")]
if static_vol is not None:
    var_models.append(("Static GARCH", static_vol, "05_var_violations_static.png"))

for label, vol, fname in var_models:
    var_line = -z99 * vol
    violations = rets_test_np < var_line

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates, rets_test_np, 0, where=(rets_test_np < 0), alpha=0.3, label="Negative return")
    ax.plot(dates, var_line, linewidth=1.2, label="99% VaR")
    ax.scatter(
        dates[violations],
        rets_test_np[violations],
        s=18,
        zorder=5,
        label=f"Violations ({violations.sum()}, {violations.mean():.1%})",
    )
    ax.set_ylabel("Log-return")
    ax.set_title(f"WTI crude — {label} 99% VaR violations")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    savefig(fname)


# ---------------------------------------------------------------------------
# Multi-step forecast: hold final (as-of-today) parameter set fixed
# ---------------------------------------------------------------------------

h = 30
with torch.no_grad():
    fcast_var, terminal_params = model.forecast_fixed_params(X_all, returns_all, h=h)

fcast_var_np = fcast_var.detach().cpu().numpy()
fcast_vol_ann = np.sqrt(fcast_var_np) * np.sqrt(252)
terminal_vol_ann = float(params["sigma"].iloc[-1] * np.sqrt(252))

print("\n[terminal parameters held fixed]")
for k, v in terminal_params.items():
    print(f"  {k:<12} {float(v.detach().cpu()):.8f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, h + 1), fcast_vol_ann, marker="o", markersize=4, linewidth=1.5, label="Forecast ann. vol")
ax.axhline(terminal_vol_ann, linestyle="--", linewidth=1, label=f"Last filtered ann. sigma = {terminal_vol_ann:.1%}")
ax.set_xlabel("Days ahead")
ax.set_ylabel("Annualised volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title(f"WTI crude {h}-day forward volatility curve — as of {all_frame.index[-1].date()}")
ax.legend()
fig.tight_layout()
savefig("06_forward_vol_curve_fixed_params.png")


# ---------------------------------------------------------------------------
# Summary stats for the writeup: pre-war baseline vs war-period regime
# ---------------------------------------------------------------------------

war_start = pd.Timestamp("2026-02-28")
pre_war = params_test[params_test.index < war_start]
war = params_test[params_test.index >= war_start]

def ann_vol(sigma: pd.Series) -> float:
    return float(sigma.mean() * np.sqrt(252))

summary = {
    "pre_war_ann_vol": ann_vol(pre_war["sigma"]),
    "war_ann_vol_mean": ann_vol(war["sigma"]),
    "war_ann_vol_peak": float(war["sigma"].max() * np.sqrt(252)),
    "peak_date": str(war["sigma"].idxmax().date()),
    "pre_war_persistence_mean": float(pre_war["persistence"].mean()),
    "war_persistence_mean": float(war["persistence"].mean()),
    "current_persistence": float(params_test["persistence"].iloc[-1]),
    "current_ann_vol": float(params_test["sigma"].iloc[-1] * np.sqrt(252)),
    "forecast_30d_ann_vol": float(fcast_vol_ann[-1]),
    "n_var_violations_war": int((rets_test_np[params_test.index >= war_start] <
                                   (-z99 * neural_vol[params_test.index >= war_start])).sum()),
}
print("\n[war-period summary]")
for k, v in summary.items():
    print(f"  {k:<26} {v}")

pd.Series(summary).to_csv(os.path.join(PLOTS_DIR, "war_period_summary.csv"))

print("\nDone.")
