#!/usr/bin/env python3
"""
VN30 Monte-Carlo Stock Simulation — Buy & Hold + Risk Metrics.

Pipeline:
  1. Fetch real VN30 OHLCV (2021-2026) via vnstock v4 (cached to CSV).
  2. Estimate per-stock daily mean return (mu) and volatility (sigma) from log returns.
  3. Simulate daily GBM paths per stock, clamping each day's move to +-7% (HOSE limit).
  4. Build an equal-weight VN30 basket index (start = 100).
  5. Trade the index with strategies on every path:
       - Buy & hold : buy day 0, hold to horizon, pay 0.15% per-side commission + 0.1% sell tax.
  6. Aggregate mean/median/best/worst, annualized return, max drawdown, Sharpe, VaR, CVaR.

Usage:
    python vn30_simulation.py [--days 252] [--paths 200] [--seed 42] [--no-chart]
"""

import os
import sys
import argparse
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnstock import Market, Listing

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")


# ─── Configuration ───────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    cache_path: str = "vn30_ohlcv_cache.csv"
    daily_limit: float = 0.07              # +-7% daily price limit (HOSE)
    commission: float = 0.0015             # 0.15% per side broker fee
    sell_tax: float = 0.001                # 0.1% securities transfer tax on sell
    n_days: int = 252                      # default horizon (1 trading year)
    n_paths: int = 200                     # Monte-Carlo path count
    seed: int = 42
    trading_days: int = 252
    var_alpha: float = 0.05                # Value-at-Risk confidence level


CFG = SimConfig()


# ─── Data Ingestion ──────────────────────────────────────────────────────────
def fetch_cached_ohlcv() -> tuple[pd.DataFrame, int]:
    if os.path.exists(CFG.cache_path):
        df = pd.read_csv(CFG.cache_path, parse_dates=["time"])
        missing = len(df)
        df["time"] = pd.to_datetime(df["time"])
        return df, missing
    df = fetch_ohlcv_data()
    df.to_csv(CFG.cache_path, index=False)
    return df, 0


def fetch_ohlcv_data(symbols: list[str]) -> pd.DataFrame:
    mrkt = Market()
    frames = []
    for sym in symbols:
        try:
            eq = mrkt.equity(symbol=sym)
            d = eq.ohlcv(start=CFG.start_date, end=CFG.end_date, interval="1D", count=2000)
        except Exception:
            continue
        d = d.copy()
        d["symbol"] = sym
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "time"]).reset_index(drop=True)


def estimate_params(df: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Per-stock daily mean log return and std -> (symbol, mu, sigma)."""
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


# ─── GBM Monte-Carlo (vectorised) ────────────────────────────────────────────
def simulate_paths(params: list[tuple], n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate equal-weight VN30 index paths (start=100). Vectorised over stocks."""
    symbols = np.array([s for s, _, _ in params])
    rets = np.array([r for _, r, _ in params], dtype=float)   # (n_stocks,)
    vols = np.array([v for _, _, v in params], dtype=float)   # (n_stocks,)
    n_stocks = len(params)

    # Generate all random shocks at once: (n_paths, n_stocks, n_days)
    z = rng.standard_normal((n_paths, n_stocks, n_days))

    # GBM: log_ret = (mu - 0.5*sigma^2) + sigma * z
    drift = (rets - 0.5 * vols ** 2)[:, None]                 # (n_stocks, 1)
    log_ret = drift + vols[:, None] * z                       # (n_paths, n_stocks, n_days)

    # Simple returns from log returns, then clamp to daily limit
    simple_ret = np.exp(log_ret) - 1.0
    simple_ret = np.clip(simple_ret, -CFG.daily_limit, CFG.daily_limit)

    # Cumulative product per stock per path
    cum = np.cumprod(1.0 + simple_ret, axis=2)                # (n_paths, n_stocks, n_days)

    # Equal-weight basket
    share = 1.0 / n_stocks
    index_paths = 100.0 * (cum * share).sum(axis=1)           # (n_paths, n_days)
    return index_paths


# ─── Strategies ──────────────────────────────────────────────────────────────
def run_buy_hold(prices: np.ndarray, capital: float = 100.0) -> np.ndarray:
    """Buy day 0, hold to end, pay fees."""
    buy_fee, sell_fee = CFG.commission, CFG.commission
    ret = prices / prices[0]
    equity = capital * ret * (1.0 - buy_fee)
    equity[-1] *= (1.0 - sell_fee - CFG.sell_tax)
    return equity


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(finals: np.ndarray, n_days: int) -> dict:
    """Aggregate cross-path metrics from final values (start=100)."""
    finals = np.asarray(finals, dtype=float)
    ann_ret = (finals / 100.0) ** (CFG.trading_days / n_days) - 1.0
    var = np.percentile(finals, CFG.var_alpha * 100)
    cvar = finals[finals <= var].mean() if np.any(finals <= var) else var
    return {
        "mean_final": float(finals.mean()),
        "median_final": float(np.median(finals)),
        "best_final": float(finals.max()),
        "worst_final": float(finals.min()),
        "ann_return": float(ann_ret.mean()),
        "win_rate": float(np.mean(finals > 100.0)),
        f"VaR_{int((1-CFG.var_alpha)*100)}%": float(var),
        f"CVaR_{int((1-CFG.var_alpha)*100)}%": float(cvar),
    }


def path_metrics(equity: np.ndarray) -> dict:
    """Drawdown + Sharpe along a single equity curve."""
    eq = np.asarray(equity, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    sharpe = (np.nan if rets.std(ddof=1) == 0 else
              (rets.mean() * CFG.trading_days) / (rets.std(ddof=1) * np.sqrt(CFG.trading_days)))
    return {"mdd": float(dd.min()), "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="VN30 Monte-Carlo GBM Simulation")
    ap.add_argument("--days", type=int, default=CFG.n_days, help="Horizon in trading days")
    ap.add_argument("--paths", type=int, default=CFG.n_paths, help="Monte-Carlo path count")
    ap.add_argument("--seed", type=int, default=CFG.seed, help="Random seed")
    ap.add_argument("--no-chart", action="store_true", help="Skip saving PNG chart")
    args = ap.parse_args()

    listing = Listing()
    symbols = listing.symbols_by_group("VN30").tolist()

    df, from_cache = fetch_cached_ohlcv()
    params = estimate_params(df)

    print(f"[data] loaded {len(df)} rows ({from_cache} from cache) | {len(params)} clean VN30 stocks")
    for sym, mu, sg in params[:5]:
        print(f"        {sym}: mu={mu:+.6f}  sigma={sg:.6f}")

    rng = np.random.default_rng(args.seed)
    idx = simulate_paths(params, args.days, args.paths, rng)

    # Run strategies on all paths
    bh_curves = np.empty((args.paths, args.days))
    bh_finals = np.empty(args.paths)

    for i in range(args.paths):
        e_bh = run_buy_hold(idx[i])
        bh_curves[i] = e_bh
        bh_finals[i] = e_bh[-1]

    m_bh = compute_metrics(bh_finals, args.days)

    # Path-level risk metrics
    dd_list = []
    sh_list = []
    for i in range(args.paths):
        pm = path_metrics(bh_curves[i])
        dd_list.append(pm["mdd"])
        if not np.isnan(pm["sharpe"]):
            sh_list.append(pm["sharpe"])

    print("\n" + "=" * 64)
    print(f"VN30 SIMULATION  |  {args.days}-day horizon | {args.paths} MC paths | seed={args.seed}")
    print("=" * 64)
    print(f"{'Metric':<28}{'Buy & Hold':>16}")
    print("-" * 44)
    print(f"{'Mean final value':<28}{m_bh['mean_final']:>16.1f}")
    print(f"{'Median final':<28}{m_bh['median_final']:>16.1f}")
    print(f"{'Best final':<28}{m_bh['best_final']:>16.1f}")
    print(f"{'Worst final':<28}{m_bh['worst_final']:>16.1f}")
    print(f"{'Annualized return':<28}{m_bh['ann_return']:>15.1%}")
    print(f"{'Sharpe (mean path)':<28}{np.nanmean(sh_list):>16.2f}")
    print(f"{'Mean max drawdown':<28}{np.mean(dd_list):>15.1%}")
    print(f"{'Win rate (end>100)':<28}{m_bh['win_rate']:>15.1%}")
    print(f"{f'VaR {int((1-CFG.var_alpha)*100)}%':<28}{m_bh[f'VaR_{int((1-CFG.var_alpha)*100)}%']:>16.1f}")
    print(f"{f'CVaR {int((1-CFG.var_alpha)*100)}%':<28}{m_bh[f'CVaR_{int((1-CFG.var_alpha)*100)}%']:>16.1f}")
    print("-" * 44)

    if not args.no_chart:
        # Chart: median-ish path
        med_bh = np.argsort(np.abs(bh_finals - np.median(bh_finals)))[0]
        med_idx = np.argsort(np.abs(idx[:, -1] - np.median(idx[:, -1])))[0]
        fig, ax = plt.subplots(figsize=(11, 6))
        days_ax = np.arange(1, args.days + 1)
        ax.plot(days_ax, idx[med_idx], color="gray", lw=0.8, label="VN30 equal-weight index")
        ax.plot(days_ax, bh_curves[med_bh], color="tab:green", lw=1.6,
                label=f"Buy & hold (median path, final {bh_finals[med_bh]:.0f})")
        ax.set_title(f"VN30 Buy & Hold — {args.days}-day GBM Monte-Carlo")
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Portfolio value (start = 100)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("vn30_sim_result.png", dpi=130)
        print("\n[chart] saved -> vn30_sim_result.png")


if __name__ == "__main__":
    main()