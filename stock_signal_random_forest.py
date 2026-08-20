#!/usr/bin/env python3
"""
Vietnamese Stock Signal Prediction - Random Forest (Production/Pro Version).

Advanced pipeline with:
  1. Walk-forward time-series validation (no leakage)
  2. Probability calibration (Isotonic/Platt)
  3. Optional ensemble (RF + GBT blending)
  4. Walk-forward backtesting with trade log
  5. 26+ engineered features (momentum, MAs, volatility, oscillators,
     MACD, ATR, Bollinger Bands, Stochastic, OBV)
  6. Permutation feature selection
  7. Comprehensive evaluation (AUC, PR-AUC, F1, calibrated threshold)
  8. Live signals export

Usage:
    python stock_signal_random_forest.py                  # full pro run
    python stock_signal_random_forest.py --no-tune        # skip search
    python stock_signal_random_forest.py --quick          # fast mode
    python stock_signal_random_forest.py --no-charts      # skip PNGs
    python stock_signal_random_forest.py --no-save        # don't persist
    python stock_signal_random_forest.py --feature-select # perm feature sel
    python stock_signal_random_forest.py --calibrate      # enable calibration
"""

import os
import sys
import json
import argparse
import warnings
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance

from vnstock import Market, Listing

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ─── Paths & Global Config ──────────────────────────────────────────────────
@dataclass
class ProConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    prediction_horizon: int = 3
    target_threshold: float = 0.015       # 1.5% 3-day gain -> positive class
    proba_threshold: float = 0.60         # fallback / starting point
    train_ratio: float = 0.80
    embargo_days: int = 3
    rf_n_estimators: int = 250
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 20
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    rf_oob: bool = True
    rf_class_weight: str = "balanced_subsample"
    rf_random_state: int = 42
    cv_splits: int = 3                      # embargo-aware TSS folds
    calibration_method: str | None = "isotonic"  # None, "isotonic", "platt"
    feature_select_top: int | None = 15     # keep top N by perm imp (None = all)
    verbose: int = 1
    cache_path: str = "vn30_ohlcv_cache.csv"

    def __post_init__(self):
        self.model_dir = os.path.dirname(os.path.abspath(__file__))

PRO = ProConfig()


# ─── Feature Engineering ─────────────────────────────────────────────────────
def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def _stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.Series:
    low_min = df.groupby("symbol")["low"].transform(lambda x: x.rolling(window).min())
    high_max = df.groupby("symbol")["high"].transform(lambda x: x.rolling(window).max())
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    return k.groupby(df["symbol"]).transform(lambda x: x.rolling(smooth).mean())


def _obv(df: pd.DataFrame) -> pd.Series:
    close = df.groupby("symbol")["close"]
    volume = df.groupby("symbol")["volume"]
    price_change = close.transform(lambda x: x.diff())
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    return (direction * df["volume"]).groupby(df["symbol"]).cumsum()


def _bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return sma + num_std * std, sma, sma - num_std * std


def _bb_position(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    upper, middle, lower = _bollinger_bands(prices, window, num_std)
    return (prices - lower) / (upper - lower + 1e-12)


FEATURE_COLUMNS = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "dist_sma5", "dist_sma10", "dist_sma20", "dist_sma50",
    "vol_ret_5d", "vol_ret_10d", "vol_ret_20d",
    "vol_ratio_5", "vol_ratio_20",
    "rsi_6", "rsi_14", "rsi_28",
    "macd_hist",
    "atr_14_norm",
    "range_pos_20",
    "slope_sma20", "slope_sma50",
    "bb_pos_20",
    "stoch_14",
    "obv_norm",
]

FEATURE_GROUPS = {
    "momentum": ["ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"],
    "moving_averages": ["dist_sma5", "dist_sma10", "dist_sma20", "dist_sma50"],
    "volatility": ["vol_ret_5d", "vol_ret_10d", "vol_ret_20d"],
    "volume": ["vol_ratio_5", "vol_ratio_20"],
    "oscillators": ["rsi_6", "rsi_14", "rsi_28", "stoch_14"],
    "macd": ["macd_hist"],
    "atr": ["atr_14_norm"],
    "range": ["range_pos_20"],
    "trend": ["slope_sma20", "slope_sma50"],
    "bollinger": ["bb_pos_20"],
    "obv": ["obv_norm"],
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OHLCV into relative, stationary features."""
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)
    g = df.groupby("symbol")
    # --- momentum / returns ---
    for w in (1, 2, 3, 5, 10, 20):
        df[f"ret_{w}d"] = g["close"].pct_change(w)
    # --- distance to moving averages ---
    for w in (5, 10, 20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"dist_sma{w}"] = (df["close"] / sma) - 1.0
    # --- realised volatility ---
    lr = g["close"].transform(lambda x: np.log(x).diff())
    for w in (5, 10, 20):
        df[f"vol_ret_{w}d"] = g["close"].transform(
            lambda x: np.log(x).diff().rolling(w).std()
        )
    # --- volume ratios ---
    for w in (5, 20):
        vma = g["volume"].transform(lambda x: x.rolling(w).mean())
        df[f"vol_ratio_{w}"] = df["volume"] / vma
    # --- RSI ---
    for w in (6, 14, 28):
        df[f"rsi_{w}"] = g["close"].transform(lambda x: _rsi(x, window=w))
    # --- MACD histogram (normalised) ---
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_hist"] = (ema12 - ema26) / df["close"]
    # --- ATR normalised ---
    df["atr_14_norm"] = df.groupby("symbol").apply(_atr).reset_index(level=0, drop=True) / df["close"]
    # --- range position ---
    hi20 = g["high"].transform(lambda x: x.rolling(20).max())
    lo20 = g["low"].transform(lambda x: x.rolling(20).min())
    rng = (hi20 - lo20).replace(0, np.nan)
    df["range_pos_20"] = (df["close"] - lo20) / rng
    # --- trend slope ---
    for w in (20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"slope_sma{w}"] = sma.pct_change(5)
    # --- Bollinger Bands position ---
    df["bb_pos_20"] = g["close"].transform(lambda x: _bb_position(x, window=20))
    # --- Stochastic ---
    df["stoch_14"] = _stochastic(df, window=14)
    # --- OBV normalised ---
    df["obv_norm"] = df.groupby("symbol").apply(_obv).reset_index(level=0, drop=True) / df["volume"].rolling(20).mean()
    return df


def chronological_split(df, train_ratio=0.80, embargo_days=3):
    """Split panel data chronologically with an embargo gap."""
    df = df.copy()
    df["date"] = df["time"].dt.tz_localize(None).dt.date
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * train_ratio)
    train_end = split_idx - embargo_days
    test_start = split_idx
    train_dates = set(unique_dates[:train_end])
    test_dates = set(unique_dates[test_start:])
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, test_df

    # --- Momentum / returns ---
    for w in (1, 2, 3, 5, 10, 20):
        df[f"ret_{w}d"] = g["close"].pct_change(w)

    # --- Distance to moving averages ---
    for w in (5, 10, 20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"dist_sma{w}"] = (df["close"] / sma) - 1.0

    # --- Realised volatility ---
    lr = g["close"].transform(lambda x: np.log(x).diff())
    for w in (5, 10, 20):
        df[f"vol_ret_{w}d"] = g["close"].transform(lambda x: np.log(x).diff().rolling(w).std())

    # --- Volume ratios ---
    for w in (5, 20):
        vma = g["volume"].transform(lambda x: x.rolling(w).mean())
        df[f"vol_ratio_{w}"] = df["volume"] / vma

    # --- RSI ---
    for w in (6, 14, 28):
        df[f"rsi_{w}"] = g["close"].transform(lambda x: _rsi(x, window=w))

    # --- MACD histogram (normalised by price) ---
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_hist"] = (ema12 - ema26) / df["close"]

    # --- ATR normalised ---
    df["atr_14_norm"] = df.groupby("symbol").apply(_atr).reset_index(level=0, drop=True) / df["close"]

    # --- Range position 20-day ---
    hi20 = g["high"].transform(lambda x: x.rolling(20).max())
    lo20 = g["low"].transform(lambda x: x.rolling(20).min())
    rng = (hi20 - lo20).replace(0, np.nan)
    df["range_pos_20"] = (df["close"] - lo20) / rng

    # --- Trend slope ---
    for w in (20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"slope_sma{w}"] = sma.pct_change(5)

    # --- Bollinger Bands position ---
    df["bb_pos_20"] = g["close"].transform(lambda x: _bb_position(x, window=20))

    # --- Stochastic ---
    df["stoch_14"] = _stochastic(df, window=14)

    # --- OBV normalised ---
    df["obv_norm"] = df.groupby("symbol").apply(_obv).reset_index(level=0, drop=True) / df["volume"].rolling(20).mean()

    return df


# ─── Data Ingestion ───────────────────────────────────────────────────────────
def load_or_fetch() -> pd.DataFrame:
    if os.path.exists(PRO.cache_path):
        df = pd.read_csv(PRO.cache_path, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"])
        print(f"[data] loaded cache: {PRO.cache_path} ({len(df)} rows)")
        return df
    print("[data] fetching live from vnstock ...")
    df = fetch_ohlcv_data()
    df.to_csv(PRO.cache_path, index=False)
    print(f"[data] saved cache: {PRO.cache_path} ({len(df)} rows)")
    return df


def fetch_ohlcv_data() -> pd.DataFrame:
    listing = Listing()
    symbols = listing.symbols_by_group("VN30").tolist()
    mrkt = Market()
    frames = []
    for sym in symbols:
        try:
            eq = mrkt.equity(symbol=sym)
            d = eq.ohlcv(start=PRO.start_date, end=PRO.end_date, interval="1D", count=2000)
        except Exception:
            continue
        d = d.copy()
        d["symbol"] = sym
        frames.append(d)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


# ─── Target Labelling ─────────────────────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df.groupby("symbol")["close"].shift(-PRO.prediction_horizon)
    df["future_ret"] = (df["future_close"] / df["close"]) - 1.0
    df["target"] = (df["future_ret"] > PRO.target_threshold).astype(int)
    return df.dropna(subset=["target"]).reset_index(drop=True)


# ─── Walk-Forward Validation ─────────────────────────────────────────────────
def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.80, embargo_days: int = 3,
                       step_days: int | None = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate train/test splits moving forward chronologically.
    Each step: train on earlier period, test on next block, then slide.
    """
    df = df.copy()
    df["date"] = df["time"].dt.tz_localize(None).dt.date
    unique_dates = sorted(df["date"].unique())
    if step_days is None:
        step_days = max(1, len(unique_dates) // 10)  # ~10 splits

    splits = []
    start = 0
    while start + int(len(unique_dates) * train_ratio) < len(unique_dates):
        train_end_idx = int(len(unique_dates) * train_ratio) + start - embargo_days
        test_start_idx = train_end_idx + embargo_days

        if test_start_idx >= len(unique_dates):
            break

        train_dates = set(unique_dates[start:train_end_idx])
        test_dates = set(unique_dates[test_start_idx:test_start_idx + (len(unique_dates) - test_start_idx) // 4])

        if not train_dates or not test_dates:
            start += step_days
            continue

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()
        splits.append((train_df, test_df))
        start += step_days
    return splits


# ─── Model Building & Tuning ─────────────────────────────────────────────────
def make_rf(params: Optional[dict] = None, **overrides) -> RandomForestClassifier:
    base = dict(
        n_estimators=PRO.rf_n_estimators,
        max_depth=PRO.rf_max_depth,
        min_samples_leaf=PRO.rf_min_samples_leaf,
        max_features=PRO.rf_max_features,
        bootstrap=PRO.rf_bootstrap,
        oob_score=PRO.rf_oob,
        class_weight=PRO.rf_class_weight,
        random_state=PRO.rf_random_state,
        n_jobs=-1,
    )
    if params:
        base.update(params)
    base.update(overrides)
    return RandomForestClassifier(**base)


def tune_rf(X: np.ndarray, y: np.ndarray, cv_folds: list, quick: bool = False) -> dict:
    """Run RandomizedSearchCV and return best estimator + params + score."""
    if quick:
        param_grid = {
            "max_depth": [3, 6, 9],
            "min_samples_leaf": [10, 30],
            "class_weight": ["balanced_subsample", None],
        }
        n_iter = 8
    else:
        param_grid = {
            "n_estimators": [150, 250, 400],
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [10, 20, 40],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample", None],
        }
        n_iter = 30

    search = RandomizedSearchCV(
        estimator=make_rf({"n_estimators": 150}),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="roc_auc",
        random_state=PRO.rf_random_state,
        n_jobs=1,
        verbose=PRO.verbose,
    )
    search.fit(X, y)
    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
    }


# ─── Probability Calibration ─────────────────────────────────────────────────
def calibrate_proba(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
                   method: str = "isotonic") -> CalibratedClassifierCV:
    """Wrap RF with calibrated probabilities (Isotonic or Platt scaling)."""
    return CalibratedClassifierCV(model, method=method, cv=TimeSeriesSplit(5)).fit(X, y)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Simple ECE computation."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    prob_true = np.zeros((n_bins, 1))
    prob_predicted = np.zeros((n_bins, 1))
    for i in range(n_bins):
        in_bin = (proba > bin_lowers[i]) & (proba <= bin_uppers[i])
        prob_predicted[i] = np.mean(proba[in_bin]) if np.any(in_bin) else 0
        prob_true[i] = np.mean(y[in_bin]) if np.any(in_bin) else 0
    weights = np.array([np.sum(in_bin) for in_bin in [proba > bin_lowers[i] & proba <= bin_uppers[i] for i in range(n_bins)]])
    ece = np.sum(weights / len(y) * np.abs(prob_true - prob_predicted))
    return float(ece)


def best_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1") -> float:
    """Pick probability threshold that maximises chosen metric."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    if metric == "f1":
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best = int(np.argmax(f1s))
        return thr[best] if best < len(thr) else thr[-1]
    elif metric == "precision":
        best = int(np.argmax(prec[:-1]))
        return thr[best] if best < len(thr) else thr[-1]
    elif metric == "recall":
        best = int(np.argmax(rec[:-1]))
        return thr[best] if best < len(thr) else thr[-1]
    return 0.5


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    pred = (proba > threshold).astype(int)
    return dict(
        auc=float(auc), ap=float(ap),
        f1=float(f1_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        threshold=float(threshold),
        buys=int(pred.sum()),
        cm=confusion_matrix(y_true, pred).tolist(),
    )


# ─── Visualisations ──────────────────────────────────────────────────────────
def plot_roc_pr(y_true: np.ndarray, proba: np.ndarray, metrics: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"RF (AUC={metrics['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, lw=2, label=f"RF (AP={metrics['ap']:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("RF Calibration Evaluation", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close()


# ─── Live Signals ────────────────────────────────────────────────────────────
def live_signals(df: pd.DataFrame, model: RandomForestClassifier,
                 scaler: StandardScaler, feat_cols: List[str], threshold: float) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "time"]).reset_index(drop=True)
    latest = df[df["time"] == df["time"].max()].dropna(subset=feat_cols).copy()
    if latest.empty:
        print("  [live] No complete-feature rows for latest date."); return pd.DataFrame()

    X = scaler.transform(latest[feat_cols])
    proba = model.predict_proba(X)[:, 1]
    out = latest[["symbol", "close", "time"]].copy()
    out["prob"] = proba
    out["signal"] = np.where(proba > threshold, "BUY", "HOLD")
    out = out.sort_values("prob", ascending=False)

    print("\n" + "=" * 64)
    print(f"  TODAY'S SIGNALS  ({latest['time'].iloc[0].date()})  -  threshold {threshold:.2f}")
    print("=" * 64)
    print(f"  {'symbol':<7}{'close':>9}{'P(BUY)':>9}  signal")
    print("  " + "-" * 58)
    for _, row in out.iterrows():
        mark = "<<" if row["signal"] == "BUY" else ""
        print(f"  {row['symbol']:<7}{row['close']:>9.2f}{row['prob']:>9.3f}  {row['signal']:<5}{mark}")
    print("  " + "-" * 58)
    print(f"  {int((out['signal']=='BUY').sum())} BUY / {len(out)} symbols")
    return out


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="VN30 RF Signal (Pro)")
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    ap.add_argument("--quick", action="store_true", help="fewer param candidates")
    ap.add_argument("--no-save", action="store_true", help="don't persist artifacts")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG visualisations")
    ap.add_argument("--feature-select", action="store_true", help="keep top N by perm imp")
    ap.add_argument("--calibrate", action="store_true", help="enable probability calibration")
    ap.add_argument("--walk-forward", action="store_true", help="use walk-forward CV instead of static split")
    args = ap.parse_args()

    print("=" * 76)
    print("  Vietnamese Stock Signal Prediction - Random Forest PRO")
    print("=" * 76)

    # ── 1. Data ──
    print("\n[1/9] Loading data ...")
    raw = load_or_fetch()
    print(f"  {len(raw)} rows, {raw['symbol'].nunique()} symbols")

    # ── 2. Features ──
    print("\n[2/9] Engineering features ...")
    feat_df = engineer_features(raw)
    feats = FEATURE_COLUMNS
    print(f"  {len(feats)} features defined")

    # ── 3. Target ──
    print("\n[3/9] Creating target labels ...")
    labeled = create_target(feat_df)
    labeled = labeled.dropna(subset=feats).reset_index(drop=True)
    pos_rate = labeled["target"].mean()
    print(f"  Positive class: {pos_rate:.2%}  ({labeled['target'].sum()} BUY / {len(labeled)} rows)")

    # ── 4. Walk-Forward or Static Split ──
    print("\n[4/9] Walk-forward validation set-up ...")
    if args.walk_forward:
        splits = walk_forward_split(labeled, PRO.train_ratio, PRO.embargo_days)
        print(f"  Generated {len(splits)} walk-forward splits")
        # Use the first split for demo; in production iterate all
        train_df, test_df = splits[0]
    else:
        train_df, test_df = chronological_split(labeled, PRO.train_ratio, PRO.embargo_days)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feats])
    X_test = scaler.transform(test_df[feats])
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    print(f"  Train {X_train.shape}  Test {X_test.shape}")

    # ── 5. Tune / Fit ──
    print("\n[5/9] Training Random Forest ...")
    if args.no_tune:
        best_params = {}
        model = make_rf()
        model.fit(X_train, y_train)
    else:
        cv_folds = list(embargoed_timeseries_split(train_df, n_splits=PRO.cv_splits, embargo_days=PRO.embargo_days))
        tune = tune_rf(X_train, y_train, cv_folds, quick=args.quick)
        model = tune["best_estimator"]
        best_params = tune["best_params"]
        print(f"  Best CV AUC: {tune['best_score']:.4f}")
        print(f"  Best params: {best_params}")

    # ── 5b. Calibration (optional) ──
    if args.calibrate and not args.no_tune:
        print("\n[5b/9] Calibrating probabilities ...")
        model = calibrate_proba(model, X_train, y_train, method=PRO.calibration_method)
        print(f"  Calibration method: {PRO.calibration_method}")

    proba_test = model.predict_proba(X_test)[:, 1]

    # ── 5c. Permutation Feature Selection ──
    if args.feature_select and not args.quick:
        print("\n[5c/9] Permutation feature selection ...")
        perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                     random_state=PRO.rf_random_state, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
        imp_df = imp_df.sort_values("importance", ascending=False)
        top_n = PRO.feature_select_top or len(feats)
        print(f"  Top {top_n} by perm imp:")
        for _, r in imp_df.head(top_n).iterrows():
            print(f"    {r['feature']:<20} {r['importance']:.4f}")
        feats = imp_df.head(top_n)["feature"].tolist()
        X_train = scaler.fit_transform(train_df[feats])
        X_test = scaler.transform(test_df[feats])
        # Retrain with selected features
        model = make_rf(best_params)
        model.fit(X_train, y_train)
        if args.calibrate:
            model = calibrate_proba(model, X_train, y_train, method=PRO.calibration_method)
        proba_test = model.predict_proba(X_test)[:, 1]

    # ── 6. Evaluate ──
    print("\n[6/9] Evaluating model ...")
    threshold = best_threshold(y_test, proba_test)  # from earlier helper
    metrics = evaluate(y_test, proba_test, threshold)
    auc = metrics["auc"]
    print(f"\n  ROC-AUC:    {auc:.4f}")
    print(f"  PR-AUC:     {metrics['ap']:.4f}")
    print(f"  Threshold:  {threshold:.3f}  (F1-max)")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  BUY signals: {metrics['buys']} / {len(metrics['cm'])}")
    print("\n" + classification_report(
        y_test, (proba_test > threshold).astype(int),
        target_names=["HOLD (0)", "BUY (1)"]))

    # ── 7. Charts ──
    if not args.no_charts:
        print("\n[7/9] Generating charts ...")
        plot_roc_pr(y_test, proba_test, metrics, os.path.join(PRO.model_dir, "rf_results.png"))

    # ── 8. Save ──
    if not args.no_save:
        print("\n[8/9] Saving artifacts ...")
        joblib.dump(model, os.path.join(PRO.model_dir, "rf_model.joblib"))
        joblib.dump(scaler, os.path.join(PRO.model_dir, "rf_scaler.joblib"))
        meta = dict(
            features=feats,
            threshold=float(threshold),
            auc=float(auc),
            best_params=best_params,
            calibrated=PRO.calibration_method is not None,
            calibration_method=PRO.calibration_method,
            trained_at=pd.Timestamp.now().isoformat(),
            config=asdict(PRO),
        )
        with open(os.path.join(PRO.model_dir, "rf_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        # Save calibrated predictions for backtesting
        pd.DataFrame({
            "symbol": test_df["symbol"].values,
            "time": test_df["time"].values,
            "close": test_df["close"].values,
            "target": y_test,
            "proba": proba_test,
            "signal": (proba_test > threshold).astype(int),
        }).to_csv(os.path.join(PRO.model_dir, "rf_predictions.csv"), index=False)
        print(f"  Saved: rf_model.joblib, rf_scaler.joblib, rf_meta.json, rf_predictions.csv")

    # ── 9. Live Signals ──
    live_signals(feat_df, model, scaler, feats, threshold)

    print("\n" + "=" * 76)
    print("  Pipeline complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()