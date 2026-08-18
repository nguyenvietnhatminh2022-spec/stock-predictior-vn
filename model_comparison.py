#!/usr/bin/env python3
"""
Multi-Model Comparison for Vietnamese Stock Signal Prediction.

This script:
  1. Fetches OHLCV data for top VN30 blue-chips (FPT, HPG, MBB) via vnstock v4.
  2. Engineers relative, stationary features (no raw prices).
  3. Labels targets using a 3-day forward return > 1.5 % threshold.
  4. Performs a strict chronological split with a 3-day embargo gap.
   5. Trains several classifiers (Logistic Regression, Random Forest,
      Gradient Boosting, SVM, KNN).
  6. Evaluates each model and saves comparison graphs as PNG images.
  7. Re-runs all steps on every execution — graphs auto-update.

Usage:
    python model_comparison.py
"""

import sys
import os
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
) 
from vnstock import Market, Listing

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

# ─── Configuration ───────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
PREDICTION_HORIZON = 3
TARGET_THRESHOLD = 0.015
PROBA_THRESHOLD = 0.60
TRAIN_RATIO = 0.80
EMBARGO_DAYS = 3
RANDOM_STATE = 42

FEATURE_COLUMNS = ["ret_1d", "ret_5d", "dist_sma20", "vol_ratio_20", "rsi_14"]

# Output directory for graphs
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Step 1: Data Ingestion ──────────────────────────────────────────────────
def fetch_ohlcv_data(symbols, start_date, end_date):
    """Fetch daily OHLCV data for each symbol via vnstock v4 Market API."""
    mrkt = Market()
    frames = []
    for sym in symbols:
        eq = mrkt.equity(symbol=sym)
        df = eq.ohlcv(start=start_date, end=end_date, interval="1D", count=2000)
        df = df.copy()
        df["symbol"] = sym
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "time"]).reset_index(drop=True)
    return combined


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────
def _rsi(prices, window=14):
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def engineer_features(df):
    """Transform raw OHLCV into relative, stationary features."""
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

    df["ret_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)

    sma20_close = df.groupby("symbol")["close"].transform(
        lambda x: x.rolling(window=20).mean()
    )
    df["dist_sma20"] = (df["close"] / sma20_close) - 1.0

    sma20_vol = df.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(window=20).mean()
    )
    df["vol_ratio_20"] = df["volume"] / sma20_vol

    df["rsi_14"] = df.groupby("symbol")["close"].transform(
        lambda x: _rsi(x, window=14)
    )

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

    print(f"\n  Train period : {min(train_dates)} -> {max(train_dates)} "
          f"({len(train_df)} rows, {len(train_dates)} dates)")
    print(f"  Embargo gap  : {unique_dates[train_end:test_start]}")
    print(f"  Test  period : {min(test_dates)} -> {max(test_dates)} "
          f"({len(test_df)} rows, {len(test_dates)} dates)")

    return train_df, test_df


def scale_features(train_df, test_df, feature_cols):
    """Fit StandardScaler on train, transform both."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    return X_train, X_test, y_train, y_test, scaler


# ─── Step 5: Model Definitions ───────────────────────────────────────────────
def get_models():
    """Return a dictionary of model names to classifier instances."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=5, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, C=1.0,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    }


# ─── Step 6: Training & Evaluation ───────────────────────────────────────────
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train each model and collect metrics."""
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train)

        # Predicted probabilities (probability of class 1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.decision_function(X_test)
            proba = (proba - proba.min()) / (proba.ptp() + 1e-9)

        buy_signals = (proba > PROBA_THRESHOLD).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, buy_signals),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, buy_signals, zero_division=0),
            "recall": recall_score(y_test, buy_signals, zero_division=0),
            "f1": f1_score(y_test, buy_signals, zero_division=0),
            "n_buy_signals": int(buy_signals.sum()),
        }
        results[name] = metrics

        print(f"    Accuracy: {metrics['accuracy']:.4f}  "
              f"ROC-AUC: {metrics['roc_auc']:.4f}  "
              f"Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}  "
              f"BUY signals: {metrics['n_buy_signals']}")

    return results


# ─── Step 7: Graph Generation ────────────────────────────────────────────────
def generate_comparison_graphs(results, output_dir):
    """Generate and save comparison graphs (auto-overwrites on each run)."""
    names = list(results.keys())
    n_models = len(names)

    acc_vals = [results[n]["accuracy"] for n in names]
    auc_vals = [results[n]["roc_auc"] for n in names]
    prec_vals = [results[n]["precision"] for n in names]
    rec_vals = [results[n]["recall"] for n in names]
    f1_vals = [results[n]["f1"] for n in names]
    buy_vals = [results[n]["n_buy_signals"] for n in names]

    # ---- Graph 1: Multi-metric bar chart ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Comparison — Vietnamese Stock Signal Prediction",
                 fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    sns.barplot(x=names, y=acc_vals, ax=ax, hue=names, palette="Blues_d", legend=False)
    ax.set_title("Accuracy (BUY/SELL decisions)")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    sns.barplot(x=names, y=auc_vals, ax=ax, hue=names, palette="Greens_d", legend=False)
    ax.set_title("ROC-AUC (Probability Ranking)")
    ax.set_ylabel("ROC-AUC")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 2]
    sns.barplot(x=names, y=prec_vals, ax=ax, hue=names, palette="Oranges_d", legend=False)
    ax.set_title("Precision (BUY signals)")
    ax.set_ylabel("Precision")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    sns.barplot(x=names, y=rec_vals, ax=ax, hue=names, palette="Purples_d", legend=False)
    ax.set_title("Recall (BUY signals)")
    ax.set_ylabel("Recall")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    sns.barplot(x=names, y=f1_vals, ax=ax, hue=names, palette="Reds_d", legend=False)
    ax.set_title("F1-Score (BUY signals)")
    ax.set_ylabel("F1-Score")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 2]
    sns.barplot(x=names, y=buy_vals, ax=ax, hue=names, palette="coolwarm", legend=False)
    ax.set_title("Number of BUY Signals (threshold=60%)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison chart: {comparison_path}")


    # ---- Graph 2: ROC-AUC vs Accuracy scatter ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(names):
        ax.scatter(acc_vals[i], auc_vals[i], s=250, zorder=5,
                   label=name, edgecolors="black", linewidth=1)

    ax.set_xlabel("Accuracy", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("Accuracy vs ROC-AUC — Model Trade-off", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.axvline(x=0.74, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "model_scatter_accuracy_auc.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter chart:   {scatter_path}")


    # ---- Graph 3: Buy Signals vs No-signal Rate ----
    fig, ax = plt.subplots(figsize=(10, 7))
    no_signal_vals = [results[n]["accuracy"] for n in names]  # base rate of HOLD dominance
    # Show BUY signal frequency vs ROC-AUC
    ax2 = ax.twinx()
    x = np.arange(n_models)
    width = 0.35
    bars1 = ax.bar(x - width/2, auc_vals, width, label="ROC-AUC", color="steelblue", alpha=0.8)
    bars2 = ax2.bar(x + width/2, buy_vals, width, label="BUY Signals", color="coral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC", color="steelblue", fontsize=12)
    ax2.set_ylabel("BUY Signal Count", color="coral", fontsize=12)
    ax.set_title("ROC-AUC vs BUY Signal Volume by Model", fontsize=15, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11)
    plt.tight_layout()
    signal_path = os.path.join(output_dir, "model_buy_signals.png")
    plt.savefig(signal_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved signal chart:   {signal_path}")


    # ---- Summary table as PNG ----
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.round(4)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    ax.set_title("Model Performance Summary", fontsize=16, fontweight="bold", pad=20)
    table = ax.table(cellText=summary_df.values,
                     rowLabels=summary_df.index,
                     colLabels=summary_df.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    plt.tight_layout()
    table_path = os.path.join(output_dir, "model_summary_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary table:  {table_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("  Multi-Model Comparison — Vietnamese Stock Signal Prediction")
    print("=" * 75)

    # Step 1 – Data Ingestion
    print("\n[1/7] Fetching VN30 symbols and OHLCV data ...")
    listing = Listing()
    symbols = listing.symbols_by_group("VN30").tolist()
    print(f"  VN30 symbols: {len(symbols)}")
    raw_df = fetch_ohlcv_data(symbols, START_DATE, END_DATE)
    print(f"  Total rows: {len(raw_df)}  ({raw_df['symbol'].nunique()} symbols)")
    print(f"  Range: {raw_df['time'].min().date()} -> {raw_df['time'].max().date()}")

    # Step 2 – Feature Engineering
    print("\n[2/7] Engineering features ...")
    feat_df = engineer_features(raw_df)
    print(f"  Features: {FEATURE_COLUMNS}")

    # Step 3 – Target Labelling
    print("\n[3/7] Creating target labels ...")
    labeled_df = create_target(feat_df)
    labeled_df = labeled_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    pos_rate = labeled_df["target"].mean()
    print(f"  Positive class ratio: {pos_rate:.2%}  "
          f"({labeled_df['target'].sum()} BUY labels)")

    # Step 4 – Chronological Split + Scaling
    print("\n[4/7] Splitting data (80/20, 3-day embargo) + scaling ...")
    train_df, test_df = chronological_split(
        labeled_df, TRAIN_RATIO, EMBARGO_DAYS
    )
    X_train, X_test, y_train, y_test, scaler = scale_features(
        train_df, test_df, FEATURE_COLUMNS
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Step 5 – Train & Evaluate Models
    print("\n[5/7] Training and evaluating models ...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Step 6 – Generate Graphs (auto-overwrites previous files)
    print("\n[7/7] Generating comparison graphs ...")
    generate_comparison_graphs(results, OUTPUT_DIR)

    # Print best model
    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n  Best model by ROC-AUC: {best_model} "
          f"(ROC-AUC={results[best_model]['roc_auc']:.4f})")

    print("\n" + "=" * 75)
    print("  All graphs saved as PNG. Re-run script to auto-update.")
    print("=" * 75)


if __name__ == "__main__":
    main()
