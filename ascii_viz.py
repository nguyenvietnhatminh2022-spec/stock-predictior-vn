#!/usr/bin/env python3
"""
ASCII Terminal Visualization — VN30 Monte-Carlo simulation.

Renders terminal-friendly charts (no matplotlib) for the VN30 simulation:
median-path equity curve (Buy & Hold) and a final-value histogram.

Usage:
    python ascii_viz.py [--days 252] [--paths 150] [--seed 42]
                        [--width 88] [--height 14]
"""

import os
import sys

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ─── ASCII chart helpers ──────────────────────────────────────────────────────
def ascii_line_chart(series, markers, title, xlabel="", ylabel="",
                     width=88, height=14):
    """Multi-series line chart. `series` = {name: 1D array}, `markers` = {name: char}."""
    n = max(len(s) for s in series.values())
    cols = min(width, n)
    xs = np.linspace(0, n - 1, cols).astype(int) if n > 1 else [0]

    allv = np.concatenate([np.asarray(s, dtype=float) for s in series.values()])
    vmin, vmax = float(allv.min()), float(allv.max())
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0

    grid = [[" "] * cols for _ in range(height)]
    for name, s in series.items():
        s = np.asarray(s, dtype=float)
        marker = markers.get(name, "*")
        for ci, xi in enumerate(xs):
            yi = s[min(xi, len(s) - 1)]
            row = int((vmax - yi) / (vmax - vmin) * (height - 1))
            row = max(0, min(height - 1, row))
            if grid[row][ci] == " ":
                grid[row][ci] = marker
            elif grid[row][ci] != marker and grid[row][ci] != "O":
                grid[row][ci] = "O"   # overlap of two series

    # y-axis labels
    fmt = lambda v: f"{v:>8.1f}"
    top, mid, bot = fmt(vmax), fmt((vmax + vmin) / 2), fmt(vmin)
    width = len(top)
    pad = " " * 2
    lines = []
    lines.append(" " * width + pad + title)
    lines.append(" " * width + pad + "-" * cols)
    for r in range(height):
        label = top if r == 0 else bot if r == height - 1 else mid if r == height // 2 else " " * width
        lines.append(label + pad + "".join(grid[r]))
    lines.append(" " * width + pad + "-" * cols)
    lines.append(" " * width + pad + (xlabel or ""))
    if ylabel:
        lines.append("  y: " + ylabel)
    lines.append("  legend: " + ",  ".join(f"{m} = {n}" for n, m in markers.items()))
    return "\n".join(lines)


def ascii_histogram(values_a, values_b, title, bins=40, width=88, height=12,
                    label_a="A", label_b="B", marker_a="*", marker_b="+"):
    """Overlay two value sets as a vertical ASCII histogram."""
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    has_b = len(vb) > 0
    lo = va.min() if not has_b else min(va.min(), vb.min())
    hi = va.max() if not has_b else max(va.max(), vb.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(va, bins=edges)
    hb, _ = np.histogram(vb, bins=edges) if has_b else (np.zeros(bins), edges)
    peak = max(ha.max(), hb.max(), 1)

    cols = bins
    grid = [[" "] * cols for _ in range(height)]
    for c in range(cols):
        ha_n = int(ha[c] / peak * (height - 1))
        hb_n = int(hb[c] / peak * (height - 1))
        for r in range(height - 1, height - 1 - max(ha_n, hb_n), -1):
            above_b = r >= height - hb_n
            above_a = r >= height - ha_n
            grid[r][c] = "O" if above_a and above_b else (marker_b if above_b else marker_a)

    ticks = [f"{lo:.0f}", f"{(lo + hi) / 2:.0f}", f"{hi:.0f}"]
    width = len(max(ticks, key=len))
    pad = " " * 2
    lines = []
    lines.append(" " * width + pad + title)
    lines.append(" " * width + pad + "-" * cols)
    for r in range(height):
        label = ticks[0] if r == height - 1 else ticks[2] if r == 0 else ticks[1] if r == height // 2 else " " * width
        lines.append(label + pad + "".join(grid[r]))
    lines.append(" " * width + pad + "-" * cols)
    lines.append(f"  legend: {marker_a} = {label_a}   {marker_b} = {label_b}")
    return "\n".join(lines)


# ─── Simulation section ───────────────────────────────────────────────────────
def run_simulation(days, paths, seed, width, height):
    import vn30_simulation as sim

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

    med_bh = int(np.argsort(np.abs(bh_finals - np.median(bh_finals)))[0])
    med_idx = int(np.argsort(np.abs(idx[:, -1] - np.median(idx[:, -1])))[0])

    print("=" * (width + 12))
    print("  VN30 MONTE-CARLO SIMULATION  —  Buy & Hold")
    print(f"  {days}-day horizon | {paths} paths | seed {seed} | {len(params)} stocks")
    print("=" * (width + 12))

    print(ascii_line_chart(
        {"index": idx[med_idx], "buy_hold": bh_curves[med_bh]},
        {"index": ".", "buy_hold": "*"},
        "Median-path portfolio value (start = 100)",
        xlabel=f"trading day 0 .. {days}", ylabel="portfolio value",
        width=width, height=height))
    print()

    print(ascii_histogram(
        bh_finals, [],
        "Final-value distribution  (buy & hold)",
        label_a="buy & hold", label_b="",
        marker_a="*", marker_b="+", width=width, height=height))

    print(f"\n  summary: buy & hold mean final {bh_finals.mean():.1f}  |  "
          f"median final {np.median(bh_finals):.1f}  (start = 100)")
    print()


def main():
    ap = argparse.ArgumentParser(description="ASCII visualization of VN30 simulation")
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--paths", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=88)
    ap.add_argument("--height", type=int, default=14)
    args = ap.parse_args()

    run_simulation(args.days, args.paths, args.seed, args.width, args.height)


if __name__ == "__main__":
    main()