import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

torch.manual_seed(42)
np.random.seed(42)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path}")

# ── Data ─────────────────────────────────────────────────────────────────────

from deepgarch.data import MarketData

data = MarketData(
    ticker="SPY",
    start="2021-06-23",
    val_start="2024-01-01",
    test_start="2025-06-01",
).load()

# ── Features ─────────────────────────────────────────────────────────────────

from deepgarch.features import (
    FeaturePipeline,
    RealizedVolatility,
    LaggedSquaredReturn,
    ReturnMomentum,
    AbsReturnMean,
)

pipeline = FeaturePipeline([
    RealizedVolatility(5),
    RealizedVolatility(22),
    LaggedSquaredReturn(1),
    LaggedSquaredReturn(5),
    ReturnMomentum(5),
    AbsReturnMean(10),
])

X_train = pipeline.fit_transform(data.train)
X_val = pipeline.transform(data.val)
X_test = pipeline.transform(data.test)

# ── Tensors ───────────────────────────────────────────────────────────────────

returns_train = torch.tensor(data.train.values, dtype=torch.float32)
returns_val = torch.tensor(data.val.values, dtype=torch.float32)
returns_test = torch.tensor(data.test.values,  dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────────────────────────

from deepgarch.models.nn import ParamNet
from deepgarch.models import GARCHNet

p, q = 1, 1

paramnet = ParamNet(
    embedding_dim=pipeline.n_features,
    hidden_dims=[32, 16],
    n_params=1 + q + p,
    dropout=0.1,
)

model = GARCHNet(
    paramnet=paramnet,
    p=p,
     q=q,
    constraint="stationary",
    max_persistence=0.999,
)

# ── Training ──────────────────────────────────────────────────────────────────

from deepgarch.train import Trainer, TrainConfig, TrainingResult
from tqdm import tqdm

class TqdmTrainer(Trainer):
    def fit(self, X_train, returns_train, X_val, returns_val):
        import time
        from pathlib import Path

        result = TrainingResult()
        checkpoint = Path(self.config.checkpoint_path)
        t0 = time.perf_counter()
        epochs_without_improvement = 0

        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"

        with tqdm(total=self.config.max_epochs, desc="training", unit="ep",
                  bar_format=bar_fmt, dynamic_ncols=True) as pbar:

            for epoch in range(self.config.max_epochs):
                train_loss = self._train_step(X_train, returns_train)
                val_loss   = self._eval_step(X_val, returns_val)
                self.scheduler.step(val_loss)

                result.train_losses.append(train_loss)
                result.val_losses.append(val_loss)

                improvement = result.best_val_loss - val_loss
                if improvement > self.config.min_delta:
                    result.best_val_loss = val_loss
                    result.best_epoch    = epoch
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), checkpoint)
                else:
                    epochs_without_improvement += 1

                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    train=f"{train_loss:.2f}",
                    val=f"{val_loss:.2f}",
                    lr=f"{lr:.2e}",
                    best=f"{result.best_val_loss:.2f}",
                    refresh=False,
                )
                pbar.update(1)

                if epochs_without_improvement >= self.config.patience:
                    pbar.write(f"  early stop at epoch {epoch + 1}  (best val={result.best_val_loss:.4f})")
                    result.stopped_early = True
                    break

        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
            checkpoint.unlink()

        result.elapsed_seconds = time.perf_counter() - t0
        return result

config = TrainConfig(
    max_epochs=500,
    learning_rate=5e-3,
    weight_decay=1e-4,
    patience=100,
    min_delta=1e-4,
    grad_clip=1.0,
    log_every=25,
)

trainer = TqdmTrainer(model, config)
result  = trainer.fit(X_train, returns_train, X_val, returns_val)

# plot: training curve
print("\n[plots]")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(result.train_losses, label="train NLL", linewidth=1.2)
ax.plot(result.val_losses,   label="val NLL",   linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("NLL")
ax.set_title("Training curve")
ax.legend()
fig.tight_layout()
savefig("01_training_curve.png")

# ── Evaluation ────────────────────────────────────────────────────────────────

from deepgarch.eval import evaluate, comparison_table, StaticGARCH

model.eval()
with torch.no_grad():
    garch_test = model.build_garch(X_test)
    neural_var = garch_test.filter(returns_test).numpy()

neural_metrics = evaluate(data.test.values, neural_var)

static_garch = StaticGARCH()
static_garch.fit(data.train)
static_var = static_garch.filter(data.test)
static_metrics = evaluate(data.test.values, static_var)

comparison_table({
    "GARCHNet": neural_metrics,
    "Static GARCH": static_metrics,
})

# plot: forecasted vol vs realised |return| (main plot)
dates      = data.test.index
realised   = np.abs(data.test.values)          # |r_t| as vol proxy
neural_vol = np.sqrt(neural_var)
static_vol = np.sqrt(static_var)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# top panel — overlay
axes[0].fill_between(dates, realised, alpha=0.25, color="grey", label="|return| (realised)")
axes[0].plot(dates, neural_vol, linewidth=1.2, color="steelblue",  label="GARCHNet")
axes[0].plot(dates, static_vol, linewidth=1.0, color="darkorange", linestyle="--", label="Static GARCH(1,1)")
axes[0].set_ylabel("Daily volatility")
axes[0].set_title("Forecasted σ vs. realised |return| — test set")
axes[0].legend()

# bottom panel — residuals (neural_vol − static_vol)
axes[1].plot(dates, neural_vol - static_vol, linewidth=0.8, color="steelblue")
axes[1].axhline(0, color="black", linewidth=0.7, linestyle="--")
axes[1].set_ylabel("GARCHNet − Static GARCH")
axes[1].set_title("Difference: GARCHNet over/under-estimates vs. static baseline")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
savefig("02_forecast_vs_realised.png")

# plot: VaR violations (99%)
alpha_var = 0.01
z99 = 2.326            # Φ⁻¹(0.01)
rets = data.test.values

for label, vol, fname in [
    ("GARCHNet", neural_vol, "03_var_violations_garchnet.png"),
    ("Static GARCH", static_vol, "04_var_violations_static.png"),
]:
    var_line   = -z99 * vol
    violations = rets < var_line

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(dates, rets, 0, where=(rets < 0), alpha=0.3, color="grey", label="Negative return")
    ax.plot(dates, var_line, linewidth=1.2, color="crimson", label="99% VaR")
    ax.scatter(dates[violations], rets[violations], color="black", s=18, zorder=5,
               label=f"Violations ({violations.sum()}, {violations.mean():.1%})")
    ax.set_ylabel("Log-return")
    ax.set_title(f"{label} — 99% VaR violations")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    savefig(fname)

# ── Multi-step forecast ───────────────────────────────────────────────────────

h = 30

with torch.no_grad():
    garch_full = model.build_garch(torch.cat([X_train, X_val, X_test]))
    all_returns = torch.cat([returns_train, returns_val, returns_test])
    fcast_var = garch_full.forecast(all_returns, h=h).numpy()

fcast_vol = np.sqrt(fcast_var) * np.sqrt(252)   # annualised
terminal_vol = np.sqrt(neural_var[-1]) * np.sqrt(252)

# plot: forward vol curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, h + 1), fcast_vol, marker="o", markersize=4, linewidth=1.5,
        color="steelblue", label="Forecast ann. vol")
ax.axhline(terminal_vol, linestyle="--", color="grey", linewidth=1,
           label=f"Last observed σ (ann.) = {terminal_vol:.1%}")
ax.set_xlabel("Days ahead")
ax.set_ylabel("Annualised volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_title(f"{h}-day forward volatility curve")
ax.legend()
fig.tight_layout()
savefig("05_forward_vol_curve.png")

print("\nDone.")