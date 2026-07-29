#!/usr/bin/env python
"""Train, evaluate, and forecast a Conditional GARCHNet model from a YAML config.

Usage:
    python run.py --config configs/spy.yaml
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from deepgarch.config import RunConfig
from deepgarch.data import MarketData
from deepgarch.eval import (
    StaticGARCH,
    comparison_table,
    evaluate,
    plot_parameter_paths,
    plot_var_violations,
    plot_volatility_comparison,
)
from deepgarch.features import FeaturePipeline
from deepgarch.models import ConditionalGARCHNet, ParamNet
from deepgarch.train.tqdm_trainer import TqdmTrainer


def to_return_tensor(frame: pd.DataFrame) -> Tensor:
    return torch.tensor(frame["returns"].to_numpy(dtype=np.float32), dtype=torch.float32)


def savefig(path: str) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def run(config: RunConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    plots_dir = config.output.dir
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------

    data = MarketData(
        ticker=config.data.ticker,
        start=config.data.start,
        end=config.data.end,
        val_start=config.data.val_start,
        test_start=config.data.test_start,
        yahoo_aux_tickers=config.data.yahoo_aux_tickers,
    ).load()
    data.summary()

    # Drop only rows where target returns are unavailable; feature NaNs are
    # handled by the pipeline.
    train_frame = data.train.loc[data.train["returns"].notna()].copy()
    val_frame = data.val.loc[data.val["returns"].notna()].copy()
    test_frame = data.test.loc[data.test["returns"].notna()].copy()
    all_frame = pd.concat([train_frame, val_frame, test_frame], axis=0)

    n_train, n_val = len(train_frame), len(val_frame)
    test_start_idx = n_train + n_val

    # ---------------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------------

    pipeline = FeaturePipeline(
        return_windows=config.features.return_windows,
        exogenous_lag=config.features.exogenous_lag,
        include_seasonality=config.features.include_seasonality,
    )
    # Fit normalization on train only, then compute rolling features over the
    # full chronological frame so validation/test starts can use prior history.
    pipeline.fit(train_frame)
    X_all = pipeline.transform(all_frame)

    X_train = X_all[:n_train]
    X_val = X_all[n_train : n_train + n_val]

    returns_all = to_return_tensor(all_frame)
    returns_train = returns_all[:n_train]
    returns_val = returns_all[n_train : n_train + n_val]
    returns_test = returns_all[test_start_idx:]

    print(f"\n[features] n_features={pipeline.n_features}")
    print("first 15 features:", pipeline.feature_names[:15])
    print(f"splits: train={len(returns_train)}, val={len(returns_val)}, test={len(returns_test)}")

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    p, q = config.model.p, config.model.q
    paramnet = ParamNet(
        embedding_dim=pipeline.n_features,
        hidden_dims=config.model.hidden_dims,
        n_params=1 + q + p,
        dropout=config.model.dropout,
    )
    model = ConditionalGARCHNet(
        paramnet=paramnet,
        p=p,
        q=q,
        constraint=config.model.constraint,
        max_persistence=config.model.max_persistence,
    )
    model_config = {
        "embedding_dim": pipeline.n_features,
        "hidden_dims": config.model.hidden_dims,
        "n_params": 1 + q + p,
        "dropout": config.model.dropout,
        "p": p,
        "q": q,
        "constraint": config.model.constraint,
        "max_persistence": config.model.max_persistence,
    }

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    trainer = TqdmTrainer(model, config.train)
    result = trainer.fit(X_train, returns_train, X_val, returns_val)
    trainer.save_artifact(market=config.output.market, model_config=model_config, artifacts_dir=plots_dir)

    print("\n[plots]")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.train_losses, label="train NLL", linewidth=1.2)
    ax.plot(result.val_losses, label="val NLL", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title(f"{config.output.market} — training curve")
    ax.legend()
    fig.tight_layout()
    savefig(os.path.join(plots_dir, "01_training_curve.png"))

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    model.eval()
    with torch.no_grad():
        # Warmed-up path: run the variance recursion over train+val+test, then
        # slice test — the t=0 seed still only depends on train returns.
        diag_all = model.diagnostics(X_all, returns_all)

    neural_var_all = diag_all["sigma2"].detach().cpu().numpy()
    neural_var = neural_var_all[test_start_idx:]
    rets_test_np = returns_test.detach().cpu().numpy()
    neural_metrics = evaluate(rets_test_np, neural_var)

    static_garch = StaticGARCH()
    static_garch.fit(train_frame["returns"])
    static_var_all = np.asarray(static_garch.filter(all_frame["returns"]))
    static_var = static_var_all[test_start_idx:]
    static_metrics = evaluate(rets_test_np, static_var)

    results = {f"{config.output.market} GARCHNet": neural_metrics, "Static GARCH": static_metrics}
    print(comparison_table(results))
    comparison_path = os.path.join(plots_dir, "comparison_metrics.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save parameter diagnostics for interpretation.
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
    params_path = os.path.join(plots_dir, "parameter_path.csv")
    params.to_csv(params_path)
    print(f"  saved -> {params_path}")

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------

    dates = test_frame.index
    events = config.output.events or None
    params_test = params.iloc[test_start_idx:]

    plot_volatility_comparison(
        rets_test_np, neural_var, static_var=static_var, index=dates, events=events,
        save_path=os.path.join(plots_dir, "02_forecast_vs_realised.png"),
    )
    print(f"  saved -> {plots_dir}/02_forecast_vs_realised.png")

    plot_parameter_paths(
        params_test["omega"], params_test["alpha"], params_test["beta"],
        index=params_test.index, events=events,
        save_path=os.path.join(plots_dir, "03_parameter_path.png"),
    )
    print(f"  saved -> {plots_dir}/03_parameter_path.png")

    plot_var_violations(
        rets_test_np, neural_var, alpha=0.01, index=dates,
        save_path=os.path.join(plots_dir, "04_var_violations_neural.png"),
    )
    print(f"  saved -> {plots_dir}/04_var_violations_neural.png")

    plot_var_violations(
        rets_test_np, static_var, alpha=0.01, index=dates,
        save_path=os.path.join(plots_dir, "05_var_violations_static.png"),
    )
    print(f"  saved -> {plots_dir}/05_var_violations_static.png")

    # ---------------------------------------------------------------------
    # Multi-step forecast: hold final parameter set fixed
    # ---------------------------------------------------------------------

    h = config.forecast.horizon
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
    ax.set_title(f"{config.output.market} — {h}-day forward volatility curve, final params held fixed")
    ax.legend()
    fig.tight_layout()
    savefig(os.path.join(plots_dir, "06_forward_vol_curve_fixed_params.png"))

    # ---------------------------------------------------------------------
    # Optional regime summary: split the test period at a given date and
    # compare average/peak volatility and persistence before vs. after.
    # ---------------------------------------------------------------------

    if config.output.regime_split:
        split_date = pd.Timestamp(config.output.regime_split)
        pre = params_test[params_test.index < split_date]
        post = params_test[params_test.index >= split_date]

        def ann_vol(sigma: pd.Series) -> float:
            return float(sigma.mean() * np.sqrt(252))

        summary = {
            "pre_split_ann_vol": ann_vol(pre["sigma"]) if len(pre) else float("nan"),
            "post_split_ann_vol_mean": ann_vol(post["sigma"]) if len(post) else float("nan"),
            "post_split_ann_vol_peak": float(post["sigma"].max() * np.sqrt(252)) if len(post) else float("nan"),
            "pre_split_persistence_mean": float(pre["persistence"].mean()) if len(pre) else float("nan"),
            "post_split_persistence_mean": float(post["persistence"].mean()) if len(post) else float("nan"),
            "current_persistence": float(params_test["persistence"].iloc[-1]),
            "current_ann_vol": float(params_test["sigma"].iloc[-1] * np.sqrt(252)),
            "forecast_horizon_ann_vol": float(fcast_vol_ann[-1]),
        }
        print(f"\n[regime summary — split at {split_date.date()}]")
        for k, v in summary.items():
            print(f"  {k:<28} {v}")
        pd.Series(summary).to_csv(os.path.join(plots_dir, "regime_summary.csv"))

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML run config.")
    args = parser.parse_args()
    run(RunConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
