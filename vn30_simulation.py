#!/usr/bin/env python3
"""
VN30 Monte-Carlo Simulation — Buy & Hold + Multiple Strategies + Risk Metrics.

Pipeline:
  1. Fetch real VN30 OHLCV (2021-2026) via vnstock v4 (cached to CSV).
  2. Estimate per-stock daily mean return (mu) and volatility (sigma) from log returns.
  3. Simulate daily GBM paths per stock, clamping each day's move to +-7% (HOSE limit).
  4. Build an equal-weight VN30 basket index (start = 100).
  5. Apply multiple strategies on every path:
       - Buy & hold : buy day 0, hold to horizon
       - Early exit   : sell after N days if price rises > X%
       - Mean reversion: sell after N days if price rises > threshold, else hold
  6. Aggregate mean/median/best/worst, annualized return, max drawdown, Sharpe, VaR, CVaR.

Usage:
    python vn30_simulation.py [--days 252] [--paths 200] [--seed 42] [--strategy BH]
    python vn30_simulation.py [--days 252] [--paths 200] [--seed 42] --strategy MR --threshold 0.05
"""

import os
import sys
import argparse
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    commission: float = 0.0015             # 0.15% per side
    sell_tax: float = 0.001                # 0.1% on sell
    n_days: int = 252                      # default horizon (1 trading year)
    n_paths: int = 200                     # Monte-Carlo path count
    seed: int = 42
    trading_days: int = 252
    var_alpha: float = 0.05                # VaR confidence level


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
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


def estimate_params(df: pd.DataFrame) -> list[tuple[str, float, float]]:
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


# ─── Vectorised GBM ──────────────────────────────────────────────────────────
def simulate_paths(params: list[tuple], n_days: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    symbols = np.array([s for s, _, _ in params])
    rets = np.array([r for _, r, _ in params], dtype=float)
    vols = np.array([v for _, _, v in params], dtype=float)
    n_stocks = len(params)

    # Shock: (n_paths, n_stocks, n_days)
    z = rng.standard_normal((n_paths, n_stocks, n_days))

    # GBM drift + diffusion
    drift = (rets - 0.5 * vols ** 2)[:, None]     # (n_stocks, 1)
    log_ret = drift + vols[:, None] * z            # (n_paths, n_stocks, n_days)

    # Simple returns, clamped to daily limit
    simple_ret = np.exp(log_ret) - 1.0
    simple_ret = np.clip(simple_ret, -CFG.daily_limit, CFG.daily_limit)

    # Cumulative return per path per stock
    cum = np.cumprod(1.0 + simple_ret, axis=2)     # (n_paths, n_stocks, n_days)

    # Equal-weight basket (start=100)
    share = 1.0 / n_stocks
    index_paths = 100.0 * (cum * share).sum(axis=1)  # (n_paths, n_days)
    return index_paths


# ─── Strategies ──────────────────────────────────────────────────────────────
def run_buy_hold(prices: np.ndarray, capital: float = 100.0) -> np.ndarray:
    """Buy day 0, hold to end, pay fees + tax."""
    buy_fee, sell_fee = CFG.commission, CFG.commission
    ret = prices / prices[0]
    equity = capital * ret * (1.0 - buy_fee)
    equity[-1] *= (1.0 - sell_fee - CFG.sell_tax)
    return equity


def run_early_exit(prices: np.ndarray, capital: float = 100.0, exit_day: int = 20,
                   exit_pct: float = 0.03) -> np.ndarray:
    """
    Sell on exit_day if price has risen by at least exit_pct from day 0.
    Otherwise hold to end.
    """
    buy_fee, sell_fee = CFG.commission, CFG.commission
    prices = np.asarray(prices, dtype=float)
    p0 = prices[0]
    equity = np.empty(len(prices))
    
    for i in range(len(prices)):
        path = prices[i]
        # Check if price at exit_day has risen enough
        if len(path) >= exit_day and (path[exit_day-1] / p0 - 1.0) >= exit_pct:
            # Sell on exit_day
            ret = path[exit_day-1] / p0
            equity[i] = capital * ret * (1.0 - buy_fee) * (1.0 - sell_fee - CFG.sell_tax)
        else:
            # Hold to end
            ret = path[-1] / p0
            equity[i] = capital * ret * (1.0 - buy_fee)
            equity[i] *= (1.0 - sell_fee - CFG.sell_tax)
    return equity


def run_mean_reversion(prices: np.ndarray, capital: float = 100.0, 
                       hold_day: int = 60, threshold: float = 0.05) -> np.ndarray:
    """
    Hold for hold_day, then sell if price has risen above threshold from purchase,
    otherwise hold to end.
    """
    buy_fee, sell_fee = CFG.commission, CFG.commission
    prices = np.asarray(prices, dtype=float)
    p0 = prices[0]
    equity = np.empty(len(prices))
    
    for i in range(len(prices)):
        path = prices[i]
        if len(path) <= hold_day:
            # Can't hold full period, just buy & hold
            ret = path[-1] / p0
            equity[i] = capital * ret * (1.0 - buy_fee) * (1.0 - sell_fee - CFG.sell_tax)
        else:
            # Check price at hold_day
            ph = path[hold_day-1]
            if (ph / p0 - 1.0) >= threshold:
                # Sell at hold_day
                ret = ph / p0
                equity[i] = capital * ret * (1.0 - buy_fee) * (1.0 - sell_fee - CFG.sell_tax)
            else:
                # Hold to end
                ret = path[-1] / p0
                equity[i] = capital * ret * (1.0 - buy_fee)
                equity[i] *= (1.0 - sell_fee - CFG.sell_tax)
    return equity


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(finals: np.ndarray, n_days: int) -> dict:
    finals = np.asarray(finals, dtype=float)
    ann_ret = (finals / 100.0) ** (CFG.trading_days / n_days) - 1.0
    var_val = np.percentile(finals, CFG.var_alpha * 100)
    cval = finals[finals <= var_val].mean() if np.any(finals <= var_val) else var_val
    return dict(
        mean_final=float(finals.mean()),
        median_final=float(np.median(finals)),
        best_final=float(finals.max()),
        worst_final=float(finals.min()),
        ann_return=float(ann_ret.mean()),
        win_rate=float(np.mean(finals > 100.0)),
        VaR_95=float(var_val),
        CVaR_95=float(cval),
    )


def path_metrics(equity: np.ndarray) -> dict:
    eq = np.asarray(equity, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    sharpe = (rets.mean() * CFG.trading_days) / (rets.std(ddof=1) * np.sqrt(CFG.trading_days)) if rets.std(ddof=1) > 0 else np.nan
    return dict(mdd=float(dd.min()), sharpe=float(sharpe) if not np.isnan(sharpe) else np.nan)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="VN30 Monte-Carlo GBM Simulation + Strategies")
    ap.add_argument("--days", type=int, default=CFG.n_days, help="Horizon in trading days")
    ap.add_argument("--paths", type=int, default=CFG.n_paths, help="Monte-Carlo path count")
    ap.add_argument("--seed", type=int, default=CFG.seed, help="Random seed")
    ap.add_argument("--strategy", type=str, default="BH",
                    choices=["BH", "early_exit", "mean_reversion"],
                    help="Strategy: BH=Buy&Hold, early_exit, mean_reversion")
    ap.add_argument("--exit-day", type=int, default=20, help="Exit day for early_exit strategy")
    ap.add_argument("--exit-pct", type=float, default=0.03, help="Exit pct for early_exit strategy")
    ap.add_argument("--hold-day", type=int, default=60, help="Hold day for mean_reversion")
    ap.add_argument("--threshold", type=float, default=0.05, help="Threshold for mean_reversion")
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

    # Run strategy on all paths
    bh_curves = np.empty((args.paths, args.days))
    bh_finals = np.empty(args.paths)

    strategy_map = {
        "BH": run_buy_hold,
        "early_exit": lambda p: run_early_exit(p, exit_day=args.exit_day, exit_pct=args.exit_pct),
        "mean_reversion": lambda p: run_mean_reversion(p, hold_day=args.hold_day, threshold=args.threshold),
    }
    strategy_fn = strategy_map[args.strategy]

    for i in range(args.paths):
        e_fn = strategy_fn(idx[i])
        bh_curves[i] = e_fn
        bh_finals[i] = e_fn[-1]

    m_bh = compute_metrics(bh_finals, args.days)

    # Path-level risk metrics
    dd_list = []
    sh_list = []
    for i in range(args.paths):
        pm = path_metrics(bh_curves[i])
        dd_list.append(pm["mdd"])
        if not np.isnan(pm["sharpe"]):
            sh_list.append(pm["sharpe"])

    print("\n" + "=" * 68)
    print(f"VN30 SIMULATION  |  {args.days}-day horizon | {args.paths} MC paths | seed={args.seed}")
    print("=" * 68)
    print(f"{'Strategy':<20}{args.strategy}")
    print("-" * 68)
    print(f"{'Mean final value':<22}{m_bh['mean_final']:>18.1f}")
    print(f"{'Median final':<22}{m_bh['median_final']:>18.1f}")
    print(f"{'Best final':<22}{m_bh['best_final']:>18.1f}")
    print(f"{'Worst final':<22}{m_bh['worst_final']:>18.1f}")
    print(f"{'Annualized return':<22}{m_bh['ann_return']:>17.1%}")
    print(f"{'Sharpe (mean path)':<22}{np.nanmean(sh_list):>18.2f}")
    print(f"{'Mean max drawdown':<22}{np.mean(dd_list):>17.1%}")
    print(f"{'Win rate (end>100)':<22}{m_bh['win_rate']:>17.1%}")
print(f"VaR 95%: {m_bh['VaR_95']:>18.1f}")
print(f"CVaR 95%: {m_bh['CVaR_95%']:>18.1f}")
    print("-" * 68)

    if not args.no_chart:
        # Chart: final value distribution + strategy line
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(bh_finals, bins=30, color="tab:blue", alpha=0.6, density=True, edgecolor='white')
        ax.axvline(100, color="black", lw=1.0, ls="--", alpha=0.5, label="Break-even")
        ax.set_title(f"VN30 {args.strategy} — {args.days}-day {args.paths}-path Monte-Carlo")
        ax.set_xlabel("Final value (start = 100)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig("vn30_sim_result.png", dpi=130)
        print(f"\n[chart] saved -> vn30_sim_result.png")
        plt.close()


if __name__ == "__main__":
    main()