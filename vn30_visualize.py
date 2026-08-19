#!/usr/bin/env python3
"""
VN30 Monte-Carlo Simulation Visualization.

Produces a multi-panel figure:
  1. Index paths fan chart (median + percentile bands).
  2. Median-path equity curve for Buy & Hold.
  3. Distribution of final values (histogram + KDE).
  4. Win-rate bar chart + risk metrics table.
  5. Drawdown distribution (optional).
  6. Annualized return distribution (optional).

Usage:
    python vn30_visualize.py [--days 252] [--paths 500] [--seed 42] [--no-save]
"""

import os
import sys
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde

import vn30_simulation as sim

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")


def parse_args():
    ap = argparse.ArgumentParser(description="VN30 Monte-Carlo Visualization")
    ap.add_argument("--days", type=int, default=sim.CFG.n_days, help="Horizon in trading days")
    ap.add_argument("--paths", type=int, default=sim.CFG.n_paths, help="Monte-Carlo path count")
    ap.add_argument("--seed", type=int, default=sim.CFG.seed, help="Random seed")
    ap.add_argument("--no-save", action="store_true", help="Don't save PNG, just show (if interactive)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load data + run simulation
    listing = sim.Listing()
    sim.CFG.SYMBOLS = listing.symbols_by_group("VN30").tolist()
    df, _ = sim.fetch_cached_ohlcv()
    params = sim.estimate_params(df)

    rng = np.random.default_rng(args.seed)
    idx = sim.simulate_paths(params, args.days, args.paths, rng)

    fees = (sim.CFG.commission, sim.CFG.commission)
    bh_curves = np.empty((args.paths, args.days))
    for p in range(args.paths):
        bh_curves[p] = sim.run_buy_hold(idx[p], 100.0)

    bh_finals = bh_curves[:, -1]
    x = np.arange(1, args.days + 1)

    # Median paths for plotting
    med_bh = int(np.argsort(np.abs(bh_finals - np.median(bh_finals)))[0])
    med_idx = int(np.argsort(np.abs(idx[:, -1] - np.median(idx[:, -1])))[0])

    # Compute risk metrics
    ann_ret = (bh_finals / 100.0) ** (sim.CFG.trading_days / args.days) - 1.0
    var_95 = np.percentile(bh_finals, 5)
    cvar_95 = bh_finals[bh_finals <= var_95].mean()

    # Drawdown for each path
    dd_curves = []
    for p in range(args.paths):
        eq = bh_curves[p]
        peak = np.maximum.accumulate(eq)
        dd_curves.append((eq - peak) / peak)
    dd_curves = np.array(dd_curves)

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # 1) Fan chart of index paths (top-left, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(percentiles)))
    for q, c in zip(percentiles, colors):
        band = np.percentile(idx, q, axis=0)
        ax1.plot(x, band, lw=1.0, alpha=0.7, color=c, label=f"P{q}")
    ax1.fill_between(x, np.percentile(idx, 5, axis=0),
                     np.percentile(idx, 95, axis=0), color="tab:blue", alpha=0.08)
    ax1.fill_between(x, np.percentile(idx, 25, axis=0),
                     np.percentile(idx, 75, axis=0), color="tab:blue", alpha=0.15)
    ax1.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_title(f"1) VN30 equal-weight index — {args.paths} paths fan chart")
    ax1.set_xlabel("Trading day")
    ax1.set_ylabel("Index (start = 100)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=7, ncol=4)

    # 2) Median-path equity curve — Buy & Hold (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x, idx[med_idx], color="gray", lw=1.0, alpha=0.8, label="VN30 index")
    ax2.plot(x, bh_curves[med_bh], color="tab:green", lw=2.0,
             label=f"Buy & hold median path (final {bh_finals[med_bh]:.0f})")
    ax2.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_title("2) Median-path equity (start = 100)")
    ax2.set_xlabel("Trading day")
    ax2.set_ylabel("Portfolio value")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    # 3) Final-value distribution — Buy & Hold (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    lo = bh_finals.min() * 0.95
    hi = bh_finals.max() * 1.05
    bins = np.linspace(lo, hi, 40)
    n, bins_used, patches = ax3.hist(bh_finals, bins=bins, color="tab:green", alpha=0.6,
                                      density=True, label=f"Buy & hold (mean {bh_finals.mean():.0f})")
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(bh_finals)
    xx = np.linspace(lo, hi, 200)
    ax3.plot(xx, kde(xx), color="darkgreen", lw=1.5, label="KDE")
    ax3.axvline(100, color="black", lw=1.0, ls="--", label="Break-even")
    ax3.axvline(bh_finals.mean(), color="tab:green", lw=1.5, ls="--", label=f"Mean {bh_finals.mean():.0f}")
    ax3.set_title("3) Distribution of final portfolio values")
    ax3.set_xlabel("Final value (start = 100)")
    ax3.set_ylabel("Density")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", fontsize=7)

    # 4) Win-rate + risk metrics table (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    win_bh = np.mean(bh_finals > 100.0) * 100.0
    metrics_text = (
        f"Win Rate (end > 100):     {win_bh:.1f}%\n"
        f"Mean Final Value:         {bh_finals.mean():.1f}\n"
        f"Median Final Value:       {np.median(bh_finals):.1f}\n"
        f"Annualized Return (mean): {ann_ret.mean():.1%}\n"
        f"Annualized Vol:           {ann_ret.std():.1%}\n"
        f"Sharpe (mean path):       {np.nanmean([sim.path_metrics(bh_curves[p])['sharpe'] for p in range(args.paths) if not np.isnan(sim.path_metrics(bh_curves[p])['sharpe'])]):.2f}\n"
        f"Mean Max Drawdown:        {np.mean([np.min(dd_curves[p]) for p in range(args.paths)]):.1%}\n"
        f"VaR 95%:                  {var_95:.1f}\n"
        f"CVaR 95%:                 {cvar_95:.1f}"
    )
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))
    ax4.set_title("4) Risk Metrics Summary")

    # 5) Drawdown fan chart (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    for q in [5, 25, 50, 75, 95]:
        band = np.percentile(dd_curves * 100, q, axis=0)
        ax5.plot(x, band, lw=1.0, alpha=0.7, label=f"P{q}")
    ax5.fill_between(x, np.percentile(dd_curves * 100, 5, axis=0),
                     np.percentile(dd_curves * 100, 95, axis=0), color="tab:red", alpha=0.08)
    ax5.fill_between(x, np.percentile(dd_curves * 100, 25, axis=0),
                     np.percentile(dd_curves * 100, 75, axis=0), color="tab:red", alpha=0.15)
    ax5.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax5.set_title("5) Drawdown fan chart (%)")
    ax5.set_xlabel("Trading day")
    ax5.set_ylabel("Drawdown (%)")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="lower left", fontsize=7, ncol=3)

    # 6) Annualized return distribution (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(ann_ret * 100, bins=30, color="tab:blue", alpha=0.6, density=True, edgecolor='white')
    ax6.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax6.axvline(ann_ret.mean() * 100, color="tab:blue", lw=1.5, ls="--", label=f"Mean {ann_ret.mean():.1%}")
    ax6.set_title("6) Annualized return distribution")
    ax6.set_xlabel("Annualized return (%)")
    ax6.set_ylabel("Density")
    ax6.grid(alpha=0.3)
    ax6.legend(loc="upper left", fontsize=7)

    # 7) Rolling Sharpe of median path (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])
    med_eq = bh_curves[med_bh]
    med_rets = np.diff(med_eq) / med_eq[:-1]
    window = 21
    if len(med_rets) >= window:
        rolling_mean = pd.Series(med_rets).rolling(window).mean()
        rolling_std = pd.Series(med_rets).rolling(window).std()
        rolling_sharpe = (rolling_mean * sim.CFG.trading_days) / (rolling_std * np.sqrt(sim.CFG.trading_days))
        # Align x with returns (x has len days, returns has len days-1)
        x_rets = x[1:]  # returns start from day 1
        valid = rolling_sharpe.notna()
        ax7.plot(x_rets[valid], rolling_sharpe[valid], color="tab:purple", lw=1.5)
    ax7.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax7.set_title("7) Rolling Sharpe (21-day, median path)")
    ax7.set_xlabel("Trading day")
    ax7.set_ylabel("Sharpe ratio")
    ax7.grid(alpha=0.3)

    # 8) Path heatmap (bottom-right, sample 50 paths)
    ax8 = fig.add_subplot(gs[2, 2])
    sample_n = min(50, args.paths)
    sample_idx = np.random.default_rng(42).choice(args.paths, sample_n, replace=False)
    im = ax8.imshow(bh_curves[sample_idx].T, aspect='auto', cmap='RdYlGn',
                    vmin=bh_finals.min(), vmax=bh_finals.max(), origin='lower')
    ax8.set_title(f"8) Sample {sample_n} equity curves")
    ax8.set_xlabel("Path index")
    ax8.set_ylabel("Trading day")
    plt.colorbar(im, ax=ax8, label="Portfolio value")

    fig.suptitle(f"VN30 Monte-Carlo — Buy & Hold   |   {args.days}-day horizon · {args.paths} paths · seed {args.seed}",
                 fontsize=14, y=0.98)

    if not args.no_save:
        out_path = "vn30_sim_visual.png"
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        print(f"[viz] saved -> {out_path}")

    if not args.no_save:
        plt.close(fig)


if __name__ == "__main__":
    main()