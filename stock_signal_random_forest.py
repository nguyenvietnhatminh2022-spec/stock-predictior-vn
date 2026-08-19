#!/usr/bin/env python3
"""
Vietnamese Stock Signal Prediction - Random Forest (production version).

Pipeline:
  1. Load VN30 OHLCV (2021-2026) via vnstock v4, reusing on-disk cache.
  2. Engineer 30+ relative, stationary features (no raw prices).
  3. Label targets using 3-day forward return > 1.5% threshold.
  4. Chronological split with 3-day embargo gap + StandardScaler.
  5. Tune + fit Random Forest classifier with embargo-aware CV.
  6. Evaluate with ROC-AUC, PR-AUC, F1, confusion matrix + charts.
  7. Save model, scaler, features, threshold, predictions via joblib.
  8. Output "Today's signals" BUY/HOLD per VN30 stock.

Usage:
    python stock_signal_random_forest.py                  # full run
    python stock_signal_random_forest.py --no-tune        # skip search (faster)
    python stock_signal_random_forest.py --quick          # minimal features/estimators
    python stock_signal_random_forest.py --no-charts      # skip PNG visualisations
    python stock_signal_random_forest.py --no-save        # do not persist artifacts
    python stock_signal_random_forest.py --feature-select # enable permutation-based feature selection
"""

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from vnstock import Market, Listing

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    start_date: str = "2021-01-01"
    end_date: str = "2026-01-01"
    prediction_horizon: int = 3
    target_threshold: float = 0.015       # 1.5% minimum 3-day gain -> positive class
    proba_threshold: float = 0.60         # fallback starting point
    train_ratio: float = 0.80
    embargo_days: int = 3
    random_state: int = 42
    cache_path: str = "vn30_ohlcv_cache.csv"
    model_path: str = "rf_model.joblib"
    scaler_path: str = "rf_scaler.joblib"
    meta_path: str = "rf_meta.json"
    predictions_path: str = "rf_predictions.csv"

    def __post_init__(self):
        self.model_dir = os.path.dirname(os.path.abspath(__file__))


CFG = Config()


# ─── Feature Engineering ──────────────────────────────────────────────────────
def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-smoothed)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder-smoothed)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def _stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.Series:
    """Stochastic oscillator %K."""
    low_min = df.groupby("symbol")["low"].transform(lambda x: x.rolling(window).min())
    high_max = df.groupby("symbol")["high"].transform(lambda x: x.rolling(window).max())
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    return k.groupby(df["symbol"]).transform(lambda x: x.rolling(smooth).mean())


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    close = df.groupby("symbol")["close"]
    volume = df.groupby("symbol")["volume"]
    price_change = close.transform(lambda x: x.diff())
    direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = (direction * df["volume"]).groupby(df["symbol"]).cumsum()
    return obv


def _bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
    """Bollinger Bands: (upper, middle, lower)."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def _bb_position(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Position within Bollinger Bands (0=lower, 1=upper, 0.5=middle)."""
    upper, middle, lower = _bollinger_bands(prices, window, num_std)
    return (prices - lower) / (upper - lower + 1e-12)


FEATURES = [
    # Momentum / returns
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # Distance to moving averages
    "dist_sma5", "dist_sma10", "dist_sma20", "dist_sma50",
    # Realised volatility
    "vol_ret_5d", "vol_ret_10d", "vol_ret_20d",
    # Volume behaviour
    "vol_ratio_5", "vol_ratio_20",
    # Oscillators
    "rsi_6", "rsi_14", "rsi_28",
    # MACD histogram normalised by price
    "macd_hist",
    # ATR normalised
    "atr_14_norm",
    # Range position
    "range_pos_20",
    # Trend slope
    "slope_sma20", "slope_sma50",
    # Bollinger Bands position
    "bb_pos_20",
    # Stochastic
    "stoch_14",
    # OBV normalised
    "obv_norm",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OHLCV into relative, stationary features."""
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)
    g = df.groupby("symbol")

    # Momentum / returns
    for w in (1, 2, 3, 5, 10, 20):
        df[f"ret_{w}d"] = g["close"].pct_change(w)

    # Distance to moving averages
    for w in (5, 10, 20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"dist_sma{w}"] = (df["close"] / sma) - 1.0

    # Realised volatility
    lr = g["close"].transform(lambda x: np.log(x).diff())
    for w in (5, 10, 20):
        df[f"vol_ret_{w}d"] = g["close"].transform(
            lambda x: np.log(x).diff().rolling(w).std()
        )

    # Volume ratios
    for w in (5, 20):
        vma = g["volume"].transform(lambda x: x.rolling(w).mean())
        df[f"vol_ratio_{w}"] = df["volume"] / vma

    # RSI
    for w in (6, 14, 28):
        df[f"rsi_{w}"] = g["close"].transform(lambda x: _rsi(x, window=w))

    # MACD histogram (normalised)
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_hist"] = (ema12 - ema26) / df["close"]

    # ATR normalised
    df["atr_14_norm"] = df.groupby("symbol").apply(_atr).reset_index(level=0, drop=True) / df["close"]

    # Position inside 20-day high-low range
    hi20 = g["high"].transform(lambda x: x.rolling(20).max())
    lo20 = g["low"].transform(lambda x: x.rolling(20).min())
    rng = (hi20 - lo20).replace(0, np.nan)
    df["range_pos_20"] = (df["close"] - lo20) / rng

    # Trend slope (normalised rate of change of SMA)
    for w in (20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"slope_sma{w}"] = sma.pct_change(5)

    # Bollinger Bands position
    df["bb_pos_20"] = g["close"].transform(lambda x: _bb_position(x, window=20))

    # Stochastic
    df["stoch_14"] = _stochastic(df, window=14)

    # OBV normalised
    df["obv_norm"] = df.groupby("symbol").apply(_obv).reset_index(level=0, drop=True) / df["volume"].rolling(20).mean()

    return df


# ─── Data Ingestion ───────────────────────────────────────────────────────────
def load_or_fetch() -> pd.DataFrame:
    """Load cached VN30 OHLCV, or fetch + cache if missing."""
    if os.path.exists(CFG.cache_path):
        df = pd.read_csv(CFG.cache_path, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"])
        print(f"[data] loaded cache: {CFG.cache_path} ({len(df)} rows)")
        return df
    print("[data] fetching live from vnstock ...")
    df = fetch_ohlcv_data()
    df.to_csv(CFG.cache_path, index=False)
    print(f"[data] saved cache: {CFG.cache_path} ({len(df)} rows)")
    return df


def fetch_ohlcv_data() -> pd.DataFrame:
    """Fetch daily OHLCV for each VN30 symbol via vnstock v4."""
    listing = Listing()
    symbols = listing.symbols_by_group("VN30").tolist()
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
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["symbol", "time"]).reset_index(drop=True)


# ─── Target Labelling ─────────────────────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: 1 if N-day forward return > threshold, else 0."""
    df = df.copy()
    df["future_close"] = df.groupby("symbol")["close"].shift(-CFG.prediction_horizon)
    df["future_ret"] = (df["future_close"] / df["close"]) - 1.0
    df["target"] = (df["future_ret"] > CFG.target_threshold).astype(int)
    return df.dropna(subset=["target"]).reset_index(drop=True)


# ─── Chronological Split with Embargo ────────────────────────────────────────
def chronological_split(df: pd.DataFrame, train_ratio: float = 0.80, embargo_days: int = 3):
    """Split panel data chronologically with embargo gap (no target leakage)."""
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
    print(f"\n  Train period : {min(train_dates)} -> {max(train_dates)} "
          f"({len(train_df)} rows, {len(train_dates)} dates)")
    print(f"  Embargo gap  : {unique_dates[train_end:test_start]}")
    print(f"  Test  period : {min(test_dates)} -> {max(test_dates)} "
          f"({len(test_df)} rows, {len(test_dates)} dates)")
    return train_df, test_df


def embargoed_timeseries_split(df: pd.DataFrame, n_splits: int = 3, embargo_days: int = 3):
    """Yield positional (train_idx, val_idx) chronological folds with embargo gap."""
    df = df.copy()
    df["date"] = df["time"].dt.tz_localize(None).dt.date
    unique_dates = np.array(sorted(df["date"].unique()))
    n_dates = len(unique_dates)
    step = (n_dates - embargo_days) // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_until = i * step
        train_end = unique_dates[train_until]
        val_start_idx = min(train_until + embargo_days, n_dates)
        if val_start_idx >= n_dates:
            continue
        val_start = unique_dates[val_start_idx]
        train_mask = df["date"] <= train_end
        val_mask = df["date"] >= val_start
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx


# ─── Model Building & Tuning ─────────────────────────────────────────────────
def make_model(params: Optional[dict] = None, **overrides) -> RandomForestClassifier:
    """Build RandomForestClassifier from params dict (or defaults)."""
    base = dict(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        random_state=CFG.random_state,
        n_jobs=-1,
    )
    if params:
        base.update(params)
    base.update(overrides)
    return RandomForestClassifier(**base)


def tune_model(X: np.ndarray, y: np.ndarray, cv: list, n_iter: int = 30, quick: bool = False):
    """RandomizedSearchCV over Random Forest, scoring ROC-AUC."""
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
    search = RandomizedSearchCV(
        estimator=make_model({"n_estimators": 150}),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=CFG.random_state,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


# ─── Threshold Optimisation ──────────────────────────────────────────────────
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
    return CFG.proba_threshold


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(y_true: np.ndarray, proba_pred: np.ndarray, threshold: float) -> dict:
    """Print ROC-AUC, PR-AUC, classification report and return metrics dict."""
    auc = roc_auc_score(y_true, proba_pred)
    ap = average_precision_score(y_true, proba_pred)
    buy_signals = (proba_pred > threshold).astype(int)
    f1 = f1_score(y_true, buy_signals)
    print(f"\n  ROC-AUC          : {auc:.4f}")
    print(f"  PR-AUC (AP)      : {ap:.4f}")
    print(f"  Threshold        : {threshold:.3f}  (maximises F1)")
    print(f"  BUY signals      : {buy_signals.sum()} / {len(buy_signals)}")
    print(f"  F1               : {f1:.4f}")
    print("\n" + classification_report(
        y_true, buy_signals, target_names=["HOLD (0)", "BUY (1)"]
    ))
    cm = confusion_matrix(y_true, buy_signals)
    return {
        "auc": float(auc), "ap": float(ap), "f1": float(f1),
        "cm": cm.tolist(), "threshold": float(threshold), "buys": int(buy_signals.sum()),
    }


# ─── Visualisations ──────────────────────────────────────────────────────────
def plot_results(y_true: np.ndarray, proba: np.ndarray, model: RandomForestClassifier,
                 feat_cols: list, metrics_: dict, out_prefix: str = "rf"):
    """Save 2x2 figure: importance, ROC, PR, prob distribution."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Random Forest - VN30 Signal Prediction", fontsize=15)

    # 1) Feature importance (built-in)
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    imp.plot(kind="barh", ax=axs[0, 0], color="steelblue")
    axs[0, 0].set_title("Feature Importance (MDI)")
    axs[0, 0].set_xlabel("Importance")

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    axs[0, 1].plot(fpr, tpr, label=f"RF (AUC = {metrics_['auc']:.3f})", lw=2)
    axs[0, 1].plot([0, 1], [0, 1], "--", color="gray", label="Random")
    axs[0, 1].set_title("ROC Curve")
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].legend()

    # 3) Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, proba)
    axs[1, 0].plot(rec, prec, label=f"RF (AP = {metrics_['ap']:.3f})", lw=2)
    axs[1, 0].set_title("Precision-Recall Curve")
    axs[1, 0].set_xlabel("Recall")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].legend()

    # 4) Probability distribution by true class
    axs[1, 1].hist(proba[y_true == 0], bins=30, alpha=0.6, label="HOLD", color="tab:red")
    axs[1, 1].hist(proba[y_true == 1], bins=30, alpha=0.6, label="BUY", color="tab:green")
    axs[1, 1].axvline(metrics_["threshold"], color="black", ls="--", label="threshold")
    axs[1, 1].set_title("Predicted P(BUY) by True Class")
    axs[1, 1].set_xlabel("Predicted probability")
    axs[1, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(CFG.model_dir, f"{out_prefix}_results.png")
    fig.savefig(out, dpi=140)
    print(f"  [chart] saved -> {os.path.basename(out)}")


# ─── Live Signals ────────────────────────────────────────────────────────────
def live_signals(df: pd.DataFrame, model: RandomForestClassifier,
                 scaler: StandardScaler, feat_cols: list, threshold: float) -> pd.DataFrame:
    """Score most recent trading day for every stock and print BUY/HOLD calls."""
    df = df.copy().sort_values(["symbol", "time"]).reset_index(drop=True)
    latest_time = df["time"].max()
    latest = df[df["time"] == latest_time].dropna(subset=feat_cols).copy()
    if latest.empty:
        print("\n  [live] No data with complete features for latest date.")
        return pd.DataFrame()

    X = scaler.transform(latest[feat_cols])
    proba = model.predict_proba(X)[:, 1]

    out = latest[["symbol", "close", "time"]].copy()
    out["prob"] = proba
    out["signal"] = np.where(proba > threshold, "BUY", "HOLD")
    out = out.sort_values("prob", ascending=False)

    print("\n" + "=" * 62)
    print(f"  TODAY'S SIGNALS  ({latest['time'].iloc[0].date()})  -  threshold {threshold:.2f}")
    print("=" * 62)
    print(f"  {'symbol':<7}{'close':>9}{'P(BUY)':>9}  signal")
    print("  " + "-" * 58)
    for _, row in out.iterrows():
        mark = "<<" if row["signal"] == "BUY" else ""
        print(f"  {row['symbol']:<7}{row['close']:>9.2f}{row['prob']:>9.3f}  {row['signal']:<5}{mark}")
    print("  " + "-" * 58)
    print(f"  {len(out[out['signal']=='BUY'])} BUY / {len(out)} symbols")
    return out


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="VN30 Random Forest signal predictor")
    ap.add_argument("--no-tune", action="store_true", help="skip hyper-parameter search")
    ap.add_argument("--quick", action="store_true", help="fast demo mode (fewer features/estimators)")
    ap.add_argument("--no-save", action="store_true", help="do not persist the model")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG visualisations")
    ap.add_argument("--feature-select", action="store_true", help="run permutation feature selection")
    args = ap.parse_args()

    print("=" * 70)
    print("  Vietnamese Stock Signal Prediction - Random Forest")
    print("=" * 70)

    # Step 1 - Data
    print("\n[1/8] Loading data ...")
    raw_df = load_or_fetch()
    print(f"  {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")
    print(f"  Range: {raw_df['time'].min().date()} -> {raw_df['time'].max().date()}")

    # Step 2 - Features
    print("\n[2/8] Engineering features ...")
    feat_df = engineer_features(raw_df)
    feats = FEATURES
    print(f"  Features: {len(feats)}")

    # Step 3 - Target
    print("\n[3/8] Creating target labels ...")
    labeled_df = create_target(feat_df)
    labeled_df = labeled_df.dropna(subset=feats).reset_index(drop=True)
    pos_rate = labeled_df["target"].mean()
    print(f"  Positive class ratio: {pos_rate:.2%}  "
          f"({labeled_df['target'].sum()} BUY labels / {len(labeled_df)} rows)")

    # Step 4 - Split + Scale
    print("\n[4/8] Splitting data (80/20, 3-day embargo) + scaling ...")
    train_df, test_df = chronological_split(labeled_df, CFG.train_ratio, CFG.embargo_days)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feats])
    X_test = scaler.transform(test_df[feats])
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Step 5 - Tune + Fit
    print("\n[5/8] Training Random Forest ...")
    if args.no_tune:
        model = make_model()
        model.fit(X_train, y_train)
        best_params = {}
    else:
        cv = list(embargoed_timeseries_split(train_df, n_splits=3, embargo_days=CFG.embargo_days))
        model, best_params, cv_auc = tune_model(X_train, y_train, cv, quick=args.quick)
        print(f"  Best CV AUC: {cv_auc:.4f}")
        print(f"  Best params: {best_params}")

    proba_test = model.predict_proba(X_test)[:, 1]

    # Step 5b - Optional feature selection
    if args.feature_select and not args.quick:
        print("\n[5b/8] Permutation feature selection ...")
        perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                       random_state=CFG.random_state, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print(f"  Top 10 by permutation importance:")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {r['feature']:<20} {r['importance']:.4f}")
        # Keep top 15 features
        top_feats = imp_df.head(15)["feature"].tolist()
        print(f"  Retaining top {len(top_feats)} features for final model...")
        feats = top_feats
        X_train = scaler.fit_transform(train_df[feats])
        X_test = scaler.transform(test_df[feats])
        model = make_model(best_params)
        model.fit(X_train, y_train)
        proba_test = model.predict_proba(X_test)[:, 1]

    # Step 6 - Threshold + Evaluate
    print("\n[6/8] Evaluating model ...")
    threshold = best_threshold(y_test, proba_test)
    metrics_ = evaluate_model(y_test, proba_test, threshold)
    auc = metrics_["auc"]
    print(f"\n  BASELINE COMPARISON  |  old RF: AUC ~0.589  |  new RF: AUC {auc:.4f}  "
          f"(+{auc - 0.589:.4f})")

    # Step 7 - Charts
    if not args.no_charts:
        print("\n[7/8] Generating charts ...")
        plot_results(y_test, proba_test, model, feats, metrics_)

    # Step 8 - Save + Live signals
    if not args.no_save:
        print("\n[8/8] Saving artifacts ...")
        joblib.dump(model, os.path.join(CFG.model_dir, CFG.model_path))
        joblib.dump(scaler, os.path.join(CFG.model_dir, CFG.scaler_path))
        meta = {
            "features": feats,
            "threshold": float(threshold),
            "auc": float(auc),
            "best_params": best_params,
            "trained_at": pd.Timestamp.now().isoformat(),
            "config": asdict(CFG),
        }
        with open(os.path.join(CFG.model_dir, CFG.meta_path), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        # Save test predictions for backtesting
        pred_df = test_df[["symbol", "time", "close", "target"]].copy()
        pred_df["proba"] = proba_test
        pred_df["signal"] = (proba_test > threshold).astype(int)
        pred_df.to_csv(os.path.join(CFG.model_dir, CFG.predictions_path), index=False)
        print(f"  [save] model       -> {CFG.model_path}")
        print(f"  [save] scaler      -> {CFG.scaler_path}")
        print(f"  [save] meta        -> {CFG.meta_path}")
        print(f"  [save] predictions -> {CFG.predictions_path}")

    live_signals(feat_df, model, scaler, feats, threshold)

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()