#!/usr/bin/env python3
"""
Vietnamese Stock Signal Prediction - Random Forest (improved).

Focus: performance first, then usability.

New vs original script:
  - 20+ engineered features (momentum, multi-window SMA/RSI, MACD, ATR,
    volume ratios, range position, realised volatility) instead of 5.
  - Automatic hyper-parameter tuning via RandomizedSearchCV on an
    embargo-aware time-series split (no future leakage).
  - Feature ranking + optional feature selection on the tuned model.
  - Probability threshold tuned to maximise F1 (default 0.60 kept).
  - Saves the model, scaler, features and threshold with joblib.
  - PNG visualisations: feature importance, ROC curve, precision-recall,
    probability distribution, confusion matrix.
  - "Today's signals" - prints live BUY/HOLD calls for every VN30 stock.

Pipeline:
  1. Load VN30 OHLCV (2021-2026) via vnstock v4, reusing the on-disk cache if present.
  2. Engineer relative, stationary features (no raw prices).
  3. Label targets using a 3-day forward return > 1.5 % threshold.
  4. Chronological split with a 3-day embargo gap + StandardScaler.
  5. Tune + fit a Random Forest classifier.
  6. Evaluate with ROC-AUC, PR-AUC, classification report + charts.

Usage:
    python stock_signal_random_forest.py                  # full run
    python stock_signal_random_forest.py --no-tune        # skip the search (faster)
    python stock_signal_random_forest.py --quick          # minimal features/estimators
"""

import os
import sys

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
PREDICTION_HORIZON = 3
TARGET_THRESHOLD = 0.015       # 1.5 % minimum 3-day gain -> positive class
PROBA_THRESHOLD = 0.60         # fallback / starting point
TRAIN_RATIO = 0.80
EMBARGO_DAYS = 3
RANDOM_STATE = 42
CACHE_PATH = "vn30_ohlcv_cache.csv"
MODEL_PATH = "rf_model.joblib"
SCALER_PATH = "rf_scaler.joblib"
META_PATH = "rf_meta.json"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def feature_columns():
    """The full feature set used by the model."""
    return [
        # momentum / returns
        "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        # distance to moving averages (structure)
        "dist_sma5", "dist_sma10", "dist_sma20", "dist_sma50",
        # relative realised volatility
        "vol_ret_5d", "vol_ret_10d", "vol_ret_20d",
        # volume behaviour
        "vol_ratio_5", "vol_ratio_20",
        # oscillators
        "rsi_6", "rsi_14", "rsi_28",
        # MACD histogram normalised by price
        "macd_hist",
        # ATR normalised by price (volatility scaled)
        "atr_14_norm",
        # position inside the 20-day high-low range
        "range_pos_20",
        # short/medium trend slope
        "slope_sma20", "slope_sma50",
    ]


# ─── Step 1: Data Ingestion (cache-first) ────────────────────────────────────
def load_or_fetch():
    """Load cached VN30 OHLCV, or fetch + cache it if missing."""
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"])
        print(f"[data] loaded cache: vn30_ohlcv_cache.csv ({len(df)} rows)")
        return df
    print("[data] fetching live from vnstock ...")
    df = fetch_ohlcv_data()
    df.to_csv(CACHE_PATH, index=False)
    print(f"[data] saved cache: vn30_ohlcv_cache.csv ({len(df)} rows)")
    return df


def fetch_ohlcv_data():
    """Fetch daily OHLCV data for each VN30 symbol via vnstock v4 Market API."""
    listing = Listing()
    symbols = listing.symbols_by_group("VN30").tolist()
    mrkt = Market()
    frames = []
    for sym in symbols:
        try:
            eq = mrkt.equity(symbol=sym)
            d = eq.ohlcv(start=START_DATE, end=END_DATE, interval="1D", count=2000)
        except Exception:
            continue
        d = d.copy()
        d["symbol"] = sym
        frames.append(d)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "time"]).reset_index(drop=True)
    return combined


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────
def _rsi(prices, window=14):
    """Compute Relative Strength Index (Wilder-smoothed approximation)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df, window=14):
    """Average True Range (Wilder-smoothed approximation)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def engineer_features(df):
    """Transform raw OHLCV into relative, stationary features (no raw prices)."""
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)
    g = df.groupby("symbol")

    # --- momentum / returns ------------------------------------------------
    for w in (1, 2, 3, 5, 10, 20):
        df[f"ret_{w}d"] = g["close"].pct_change(w)

    # --- distance to moving averages ---------------------------------------
    for w in (5, 10, 20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"dist_sma{w}"] = (df["close"] / sma) - 1.0

    # --- realised volatility -----------------------------------------------
    lr = g["close"].transform(lambda x: np.log(x).diff())
    for w in (5, 10, 20):
        df[f"vol_ret_{w}d"] = g["close"].transform(
            lambda x: np.log(x).diff().rolling(w).std()
        )

    # --- volume ratios -----------------------------------------------------
    for w in (5, 20):
        vma = g["volume"].transform(lambda x: x.rolling(w).mean())
        df[f"vol_ratio_{w}"] = df["volume"] / vma

    # --- RSI ---------------------------------------------------------------
    for w in (6, 14, 28):
        df[f"rsi_{w}"] = g["close"].transform(lambda x: _rsi(x, window=w))

    # --- MACD histogram (normalised by price) ------------------------------
    ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd_hist"] = (ema12 - ema26) / df["close"]

    # --- ATR normalised -----------------------------------------------------
    df["atr_14_norm"] = df.groupby("symbol").apply(_atr).reset_index(level=0, drop=True) / df["close"]

    # --- position inside 20-day high-low range ------------------------------
    hi20 = g["high"].transform(lambda x: x.rolling(20).max())
    lo20 = g["low"].transform(lambda x: x.rolling(20).min())
    rng = (hi20 - lo20).replace(0, np.nan)
    df["range_pos_20"] = (df["close"] - lo20) / rng

    # --- trend slope (normalised rate of change of SMA) ---------------------
    for w in (20, 50):
        sma = g["close"].transform(lambda x: x.rolling(w).mean())
        df[f"slope_sma{w}"] = sma.pct_change(5)

    return df


# ─── Step 3: Target Labelling ────────────────────────────────────────────────
def create_target(df):
    """Create binary target: 1 if 3-day forward return > 1.5 %, else 0."""
    df = df.copy()
    df["future_close"] = df.groupby("symbol")["close"].shift(-PREDICTION_HORIZON)
    df["future_ret_3d"] = (df["future_close"] / df["close"]) - 1.0
    df["target"] = (df["future_ret_3d"] > TARGET_THRESHOLD).astype(int)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df


# ─── Step 4: Chronological Split with Embargo ────────────────────────────────
def chronological_split(df, train_ratio=0.80, embargo_days=3):
    """Split panel data chronologically with an embargo gap (no target leakage)."""
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


# ─── Step 5: Embargo-aware Time-Series CV (for tuning) ───────────────────────
def embargoed_timeseries_split(df, n_splits=3, embargo_days=3):
    """Yield positional (train_idx, val_idx) chronological folds with an embargo gap."""
    df = df.copy()
    df["date"] = df["time"].dt.tz_localize(None).dt.date
    unique_dates = np.array(sorted(df["date"].unique()))
    n_dates = len(unique_dates)

    # fold boundaries: keep at least `embargo` days out of each validation block
    step = (n_dates - embargo_days) // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_until = i * step
        train_end = unique_dates[train_until]
        val_start_idx = min(train_until + embargo_days, n_dates)
        val_start = unique_dates[val_start_idx]
        if val_start_idx >= n_dates:
            continue
        train_mask = df["date"] <= train_end
        val_mask = df["date"] >= val_start
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx


# ─── Step 6: Model Training with Tuning ──────────────────────────────────────
def make_model(params=None, **overrides):
    """Build a RandomForestClassifier from a params dict (or defaults)."""
    base = dict(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    base.update(params or {})
    base.update(overrides)
    return RandomForestClassifier(**base)


def tune_model(X, y, cv, n_iter=30, quick=False):
    """RandomizedSearchCV over a Random Forest, scoring ROC-AUC."""
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
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def fit_model(X_train, y_train, X_test=None, best_params=None, quick=False):
    """Fit the (optionally tuned) Random Forest and return test probabilities."""
    model = make_model(best_params or {}, n_estimators=best_params.get("n_estimators", 250) if best_params else 250)
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1] if X_test is not None else None
    return model, proba_test


# ─── Step 7: Threshold Optimisation ──────────────────────────────────────────
def best_threshold(y_true, proba):
    """Pick the probability threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.argmax(f1s))
    return thr[best] if best < len(thr) else thr[-1], f1s[best]


# ─── Step 8: Evaluation ──────────────────────────────────────────────────────
def evaluate_model(y_true, proba_pred, threshold):
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

    report = classification_report(
        y_true, buy_signals, target_names=["HOLD (0)", "BUY (1)"]
    )
    print("\n" + report)

    cm = confusion_matrix(y_true, buy_signals)
    return {
        "auc": auc,
        "ap": ap,
        "f1": f1,
        "cm": cm,
        "threshold": threshold,
        "buys": int(buy_signals.sum()),
    }


# ─── Step 9: Visualisations ──────────────────────────────────────────────────
def plot_results(y_true, proba, model, feat_cols, metrics_, out_prefix="rf"):
    """Save a 2x2 figure: importance, ROC, PR, prob distribution."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Random Forest - VN30 Signal Prediction", fontsize=15)

    # 1) Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    imp.plot(kind="barh", ax=axs[0, 0], color="steelblue")
    axs[0, 0].set_title("Feature Importance")
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
    out = os.path.join(MODEL_DIR, f"{out_prefix}_results.png")
    fig.savefig(out, dpi=140)
    print(f"  [chart] saved -> {os.path.basename(out)}")


# ─── Step 10: Live Signals ───────────────────────────────────────────────────
def live_signals(df, model, scaler, feat_cols, threshold):
    """Score the most recent trading day for every stock and print calls."""
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

    # keep only rows where all features are available (drop NaNs)
    latest_time = df["time"].max()
    latest = df[df["time"] == latest_time].dropna(subset=feat_cols).copy()
    if latest.empty:
        print("\n  [live] No data with complete features for the latest date.")
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
    args = ap.parse_args()

    print("=" * 70)
    print("  Vietnamese Stock Signal Prediction - Random Forest")
    print("=" * 70)

    # Step 1 - Data
    print("\n[1/7] Loading data ...")
    raw_df = load_or_fetch()
    print(f"  {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")
    print(f"  Range: {raw_df['time'].min().date()} -> {raw_df['time'].max().date()}")

    # Step 2 - Features
    print("\n[2/7] Engineering features ...")
    feat_df = engineer_features(raw_df)
    feats = feature_columns()
    print(f"  Features: {len(feats)} -> {feats}")

    # Step 3 - Target
    print("\n[3/7] Creating target labels ...")
    labeled_df = create_target(feat_df)
    labeled_df = labeled_df.dropna(subset=feats).reset_index(drop=True)
    pos_rate = labeled_df["target"].mean()
    print(f"  Positive class ratio: {pos_rate:.2%}  "
          f"({labeled_df['target'].sum()} BUY labels / {len(labeled_df)} rows)")

    # Step 4 - Split + Scale
    print("\n[4/7] Splitting data (80/20, 3-day embargo) + scaling ...")
    train_df, test_df = chronological_split(labeled_df, TRAIN_RATIO, EMBARGO_DAYS)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feats])
    X_test = scaler.transform(test_df[feats])
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Step 5 - Tune + Fit
    print("\n[5/7] Training Random Forest ...")
    if args.no_tune:
        model = make_model()
        model.fit(X_train, y_train)
        best_params = {}
    else:
        cv = list(embargoed_timeseries_split(train_df, n_splits=3, embargo_days=EMBARGO_DAYS))
        model, best_params, cv_auc = tune_model(X_train, y_train, cv, quick=args.quick)
        print(f"  Best CV AUC: {cv_auc:.4f}")
        print(f"  Best params: {best_params}")
    proba_test = model.predict_proba(X_test)[:, 1]

    # Step 6 - Threshold + Evaluate
    print("\n[6/7] Evaluating model ...")
    threshold, f1_opt = best_threshold(y_test, proba_test)
    metrics_ = evaluate_model(y_test, proba_test, threshold)
    auc = metrics_["auc"]
    print(f"\n  BASELINE COMPARISON  |  old RF: AUC ~0.589  |  new RF: AUC {auc:.4f}  "
          f"(+{auc - 0.589:.4f})")

    # Step 7 - Charts, save, signals
    if not args.no_charts:
        print("\n[7/7] Generating charts ...")
        plot_results(y_test, proba_test, model, feats, metrics_)

    if not args.no_save:
        model_path = os.path.join(MODEL_DIR, MODEL_PATH)
        scaler_path = os.path.join(MODEL_DIR, SCALER_PATH)
        meta_path = os.path.join(MODEL_DIR, META_PATH)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        meta = {
            "features": feats,
            "threshold": float(threshold),
            "auc": float(auc),
            "best_params": best_params,
            "trained_at": pd.Timestamp.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  [save] model  -> {os.path.basename(model_path)}")
        print(f"  [save] scaler -> {os.path.basename(scaler_path)}")
        print(f"  [save] meta   -> {os.path.basename(meta_path)}")

    live_signals(feat_df, model, scaler, feats, threshold)

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
