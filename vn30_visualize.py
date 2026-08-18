#!/usr/bin/env python3
"""
VN30 Simulation Visualization.

Reuses the Monte-Carlo engine from vn30_simulation.py and produces a
multi-panel figure:
  1. Index paths fan chart (median + percentile bands).
  2. Median-path equity curve for Buy & Hold.
  3. Distribution of final values (histogram) for Buy & Hold.
  4. Win-rate (P(final > 100)).

Usage:
    python vn30_visualize.py [--days 252] [--paths 500] [--seed 42]
"""

import os
import sys

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import vn30_simulation as sim

warnings.filterwarnings("ignore")


def main():
    days = sim.N_DAYS
    paths = sim.N_PATHS
    seed = sim.SEED
    i = 1
    while i < len(sys.argv):
        a, i = sys.argv[i], i + 1
        if a == "--days":
            days = int(sys.argv[i]); i += 1
        elif a == "--paths":
            paths = int(sys.argv[i]); i += 1
        elif a == "--seed":
            seed = int(sys.argv[i]); i += 1

    # ─── Load data + run the simulation ───────────────────────────────────────
    listing = sim.Listing()
    sim.SYMBOLS = listing.symbols_by_group("VN30").tolist()
    df, _ = sim.fetch_cached_ohlcv()
    params = sim.estimate_params(df)

    rng = np.random.default_rng(seed)
    idx = sim.simulate_paths(params, days, paths, rng)

    fees = (sim.COMMISSION, sim.COMMISSION)
    bh_curves = np.empty((paths, days))
    for p in range(paths):
        bh_curves[p] = sim.run_buy_hold(idx[p], 100.0, fees)

    bh_finals = bh_curves[:, -1]

    x = np.arange(1, days + 1)

    # ─── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18)

    # 1) Fan chart of index paths
    ax1 = fig.add_subplot(gs[0, 0])
    for q in (5, 25, 50, 75, 95):
        band = np.percentile(idx, q, axis=0)
        ax1.plot(x, band, lw=1.0, alpha=0.75,
                 color="tab:blue", label=f"P{q}")
    ax1.fill_between(x, np.percentile(idx, 5, axis=0),
                     np.percentile(idx, 95, axis=0),
                     color="tab:blue", alpha=0.10)
    ax1.fill_between(x, np.percentile(idx, 25, axis=0),
                     np.percentile(idx, 75, axis=0),
                     color="tab:blue", alpha=0.18)
    ax1.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_title(f"1) VN30 equal-weight index — {paths} paths fan chart")
    ax1.set_xlabel("Trading day"); ax1.set_ylabel("Index (start = 100)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    # 2) Median-path equity curve — Buy & Hold
    ax2 = fig.add_subplot(gs[0, 1])
    med_bh = int(np.argsort(np.abs(bh_finals - np.median(bh_finals)))[0])
    med_idx = int(np.argsort(np.abs(idx[:, -1] - np.median(idx[:, -1])))[0])
    ax2.plot(x, idx[med_idx], color="gray", lw=1.0, alpha=0.8,
             label="VN30 index")
    ax2.plot(x, bh_curves[med_bh], color="tab:green", lw=2.0,
             label=f"Buy & hold median path (final {bh_finals[med_bh]:.0f})")
    ax2.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_title("2) Median-path equity (start = 100)")
    ax2.set_xlabel("Trading day"); ax2.set_ylabel("Portfolio value")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    # 3) Final-value distribution — Buy & Hold
    ax3 = fig.add_subplot(gs[1, 0])
    lo = bh_finals.min() * 0.95
    hi = bh_finals.max() * 1.05
    bins = np.linspace(lo, hi, 40)
    ax3.hist(bh_finals, bins=bins, color="tab:green", alpha=0.6,
             label=f"Buy & hold (mean {bh_finals.mean():.0f})")
    ax3.axvline(100, color="black", lw=1.0, ls="--")
    ax3.set_title("3) Distribution of final portfolio values")
    ax3.set_xlabel("Final value (start = 100)"); ax3.set_ylabel("Paths")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper left", fontsize=8)

    # 4) Win-rate bar chart — Buy & Hold
    ax4 = fig.add_subplot(gs[1, 1])
    win_bh = np.mean(bh_finals > 100.0) * 100.0
    bars = ax4.bar(["Buy & hold"],
                   [win_bh],
                   color=["tab:green"], alpha=0.8)
    ax4.bar_label(bars, fmt="%.1f%%")
    ax4.axhline(50, color="black", lw=0.8, ls="--", alpha=0.5)
    ax4.set_ylim(0, 105)
    ax4.set_title("4) Win rate (end > 100)")
    ax4.set_ylabel("% of paths")
    ax4.grid(alpha=0.3, axis="y")

    fig.suptitle(f"VN30 Monte-Carlo — Buy & Hold   |   {days}-day horizon · {paths} paths · seed {seed}",
                 fontsize=14, y=0.97)
    fig.savefig("vn30_sim_visual.png", dpi=140)
    print(f"[viz] saved -> vn30_sim_visual.png")


if __name__ == "__main__":
    main()