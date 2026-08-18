#!/usr/bin/env python3
"""
VN30 Monte-Carlo Stock Simulation — Buy & Hold.

Pipeline:
  1. Fetch real VN30 OHLCV (2021-2026) via vnstock v4 (cached to CSV).
  2. Estimate per-stock daily mean return (mu) and volatility (sigma) from log returns.
  3. Simulate daily GBM paths per stock, clamping each day's move to +-7% (HOSE limit).
  4. Build an equal-weight VN30 basket index (start = 100).
  5. Trade the index with buy & hold on every path:
       - Buy & hold : buy day 0, hold to the horizon, pay 0.15% per-side
                      commission + 0.1% sell tax.
  6. Aggregate mean/median/best/worst, annualized return, max drawdown, Sharpe.

Usage:
    python vn30_simulation.py [--days 252] [--paths 200] [--seed 42]
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
import pandas as pd

from vnstock import Market, Listing

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
SYMBOLS = None                       # resolved from VN30 group
CACHE_PATH = "vn30_ohlcv_cache.csv"

# Trading / settlement parameters — Vietnam (HOSE)
DAILY_LIMIT = 0.07                   # +-7% daily price limit
COMMISSION = 0.0015                  # 0.15% per side broker fee
SELL_TAX = 0.001                     # 0.1% securities transfer tax on sell value

N_DAYS = 252                         # default horizon (1 trading year)
N_PATHS = 200                        # Monte-Carlo path count
SEED = 42
TRADING_DAYS = 252


# ─── Step 1: Data Ingestion (cached) ─────────────────────────────────────────
def fetch_cached_ohlcv():
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["time"])
        missing = len(df)
        df["time"] = pd.to_datetime(df["time"])
        return df, missing
    df = fetch_ohlcv_data()
    df.to_csv(CACHE_PATH, index=False)
    return df, 0


def fetch_ohlcv_data():
    mrkt = Market()
    frames = []
    for sym in SYMBOLS:
        try:
            eq = mrkt.equity(symbol=sym)
            d = eq.ohlcv(start=START_DATE, end=END_DATE, interval="1D", count=2000)
        except Exception:
            continue
        d = d.copy()
        d["symbol"] = sym
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "time"]).reset_index(drop=True)
    return out


def estimate_params(df):
    """Per-stock daily mean log return and std -> (mu, sigma)."""
    rows = []
    for sym, g in df.groupby("symbol"):
        p = g["close"].values
        if len(p) < 60:
            continue
        lr = np.diff(np.log(p))
        rows.append((sym, float(lr.mean()), float(lr.std())))
    if not rows:
        raise RuntimeError("No clean price series available.")
    return rows


# ─── Step 2: GBM Monte-Carlo ─────────────────────────────────────────────────
def simulate_paths(params, n_days, n_paths, rng):
    """Simulate equal-weight VN30 index paths (start=100), one BM path per stock."""
    rets = np.array([r for _, r, _ in params], dtype=float)        # daily log-mean
    vols = np.array([v for _, _, v in params], dtype=float)        # daily log-std
    n_stocks = len(params)

    index_paths = np.empty((n_paths, n_days), dtype=float)
    share = 1.0 / n_stocks

    for i in range(n_paths):
        z = rng.standard_normal((n_stocks, n_days))
        drift = (rets - 0.5 * vols ** 2)
        log_ret = drift[:, None] + vols[:, None] * z
        s = np.exp(log_ret)
        m = np.maximum(s - 1.0, -DAILY_LIMIT)
        s = np.where(s - 1.0 > DAILY_LIMIT, 1.0 + DAILY_LIMIT, 1.0 + m)
        cum = np.cumprod(s, axis=1)
        weighted = cum * share
        index_paths[i] = 100.0 * weighted.sum(axis=0)

    return index_paths


# ─── Step 3: Buy & Hold strategy ─────────────────────────────────────────────
def run_buy_hold(prices, capital, fees):
    buy_fee, sell_fee = fees
    ret = prices / prices[0]
    equity = capital * ret * (1.0 - buy_fee)
    equity[-1] *= (1.0 - sell_fee - SELL_TAX)
    return equity


# ─── Step 5: Metrics ─────────────────────────────────────────────────────────
def metrics(finals, n_days):
    """Aggregate cross-path metrics from an array of final values (start=100)."""
    finals = np.asarray(finals, dtype=float)
    ann_ret = (finals / 100.0) ** (TRADING_DAYS / n_days) - 1.0
    return {
        "final": float(finals.mean()),
        "ann_ret": float(ann_ret.mean()),
        "win": float(np.mean(finals > 100.0)),
    }


def path_metrics(equity):
    """Drawdown + Sharpe along a single equity curve."""
    eq = np.asarray(equity, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    sharpe = (rets.mean() * TRADING_DAYS) / (rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if rets.std(ddof=1) > 0 else np.nan
    return {"mdd": float(dd.min()), "sharpe": sharpe}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    global SYMBOLS
    days = N_DAYS
    paths = N_PATHS
    seed = SEED
    i = 1
    while i < len(sys.argv):
        a, i = sys.argv[i], i + 1
        if a == "--days":
            days = int(sys.argv[i]); i += 1
        elif a == "--paths":
            paths = int(sys.argv[i]); i += 1
        elif a == "--seed":
            seed = int(sys.argv[i]); i += 1

    listing = Listing()
    SYMBOLS = listing.symbols_by_group("VN30").tolist()
    df, from_cache = fetch_cached_ohlcv()
    params = estimate_params(df)

    print(f"[data] loaded {len(df)} rows ({from_cache} from cache) | {len(params)} clean VN30 stocks")
    print(f"[param] best/sigma examples:")
    for sym, mu, sg in params[:5]:
        print(f"        {sym}: mu={mu:+.6f}  sigma={sg:.6f}")

    rng = np.random.default_rng(seed)
    idx = simulate_paths(params, days, paths, rng)

    fees = (COMMISSION, COMMISSION)
    bh_finals = np.empty(paths)
    bh_curves = []

    for i in range(paths):
        e_bh = run_buy_hold(idx[i], 100.0, fees)
        bh_finals[i] = e_bh[-1]
        bh_curves.append(e_bh)

    m_bh = metrics(bh_finals, days)

    dd_bh = []
    sh_bh = []
    for i in range(paths):
        pm_bh = path_metrics(bh_curves[i])
        dd_bh.append(pm_bh["mdd"])
        sh_bh.append(pm_bh["sharpe"])
    dd_bh_mean = np.mean(dd_bh)
    sh_bh_mean = np.nanmean(sh_bh)

    print("\n" + "=" * 60)
    print(f"VN30 SIMULATION  |  {days}-day horizon | {paths} Monte-Carlo paths | seed={seed}")
    print("=" * 60)
    print(f"{'Metric':<24}{'Buy & Hold':>16}")
    print("-" * 40)
    print(f"{'Mean final value':<24}{m_bh['final']:>16.1f}")
    print(f"{'Median final':<24}{np.median(bh_finals):>16.1f}")
    print(f"{'Best path':<24}{bh_finals.max():>16.1f}")
    print(f"{'Worst path':<24}{bh_finals.min():>16.1f}")
    print(f"{'Annualized return':<24}{m_bh['ann_ret']:>15.1%}")
    print(f"{'Sharpe (mean path)':<24}{sh_bh_mean:>16.2f}")
    print(f"{'Mean max drawdown':<24}{dd_bh_mean:>15.1%}")
    print(f"{'Win rate (end>100)':<24}{m_bh['win']:>15.1%}")
    print("-" * 40)

    # Chart: median-ish path (closest to median final)
    med_bh = np.argsort(np.abs(bh_finals - np.median(bh_finals)))[0]
    med_idx = np.argsort(np.abs(idx[:, -1] - np.median(idx[:, -1])))[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    days_ax = np.arange(1, days + 1)
    ax.plot(days_ax, idx[med_idx], color="gray", lw=0.8, label="VN30 equal-weight index")
    ax.plot(days_ax, bh_curves[med_bh], color="tab:green", lw=1.6, label=f"Buy & hold (median path, final {bh_finals[med_bh]:.0f})")
    ax.set_title(f"VN30 Buy & Hold — {days}-day GBM Monte-Carlo")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio value (start = 100)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("vn30_sim_result.png", dpi=130)
    print("\n[chart] saved -> vn30_sim_result.png")


if __name__ == "__main__":
    main()