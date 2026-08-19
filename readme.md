# VN30 Stock Signal Prediction & Monte-Carlo Simulation

A production-ready Python toolkit for analyzing the Vietnamese VN30 index using machine learning and Monte-Carlo methods.

## What This Project Does

1. **ML Signal Prediction** — Predicts whether a VN30 stock will rise >1.5% in 3 days using a tuned Random Forest with 26 engineered features
2. **Monte-Carlo Simulation** — Simulates VN30 equal-weight basket forward with GBM respecting HOSE rules (±7% daily limit, fees, taxes)
3. **Rich Visualizations** — 8-panel dashboard charts, ROC/PR curves, feature importance, risk metrics tables

## Quick Start

```powershell
# 1. Setup (one-time)
python -m venv ~/.venv
& "$HOME\.venv\Scripts\Activate.ps1"
python -m pip install -U pip
python -m pip install -U vnstock vnai numpy pandas scikit-learn matplotlib scipy pyyaml

# 2. Run full ML training (with hyperparameter tuning)
& "$HOME\.venv\Scripts\python.exe" stock_signal_random_forest.py

# 3. Or fast background training
& "$HOME\.venv\Scripts\python.exe" train_rf.py --daemon --notify

# 4. Run Monte-Carlo simulation
& "$HOME\.venv\Scripts\python.exe" vn30_simulation.py --days 252 --paths 200

# 5. Generate 8-panel visualization dashboard
& "$HOME\.venv\Scripts\python.exe" vn30_visualize.py --days 252 --paths 500
```

## Scripts Overview

### `stock_signal_random_forest.py` — Main ML Pipeline
**Purpose:** Train & evaluate Random Forest to predict 3-day forward returns >1.5%

**Features (26):**
- Momentum: `ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`
- MA distance: `dist_sma5`, `dist_sma10`, `dist_sma20`, `dist_sma50`
- Realized volatility: `vol_ret_5d`, `vol_ret_10d`, `vol_ret_20d`
- Volume: `vol_ratio_5`, `vol_ratio_20`
- Oscillators: `rsi_6`, `rsi_14`, `rsi_28`, `stoch_14`
- MACD: `macd_hist`
- ATR: `atr_14_norm`
- Position: `range_pos_20`, `bb_pos_20`
- Trend: `slope_sma20`, `slope_sma50`
- Volume flow: `obv_norm`

**Key Features:**
- Embargo-aware time-series CV (3-day gap, no leakage)
- RandomizedSearchCV over 30 param combinations × 3 folds
- F1-optimized probability threshold
- Permutation importance feature selection (`--feature-select`)
- Saves: `rf_model.joblib`, `rf_scaler.joblib`, `rf_meta.json`, `rf_predictions.csv`

**Usage:**
```powershell
# Full run with tuning & charts
python stock_signal_random_forest.py

# Fast mode (no tuning, fewer estimators)
python stock_signal_random_forest.py --quick

# Skip tuning, use defaults
python stock_signal_random_forest.py --no-tune

# Skip charts
python stock_signal_random_forest.py --no-charts

# Enable permutation feature selection
python stock_signal_random_forest.py --feature-select
```

### `train_rf.py` — Background Training Runner
**Purpose:** Wrapper for unattended/background training with logging & notifications

**Features:**
- Daemon mode (`--daemon`) — runs detached in background
- YAML config file support (`--config config.yaml`)
- Desktop notifications on completion (`--notify`)
- Timestamped log files
- Cross-platform (Windows `start /B`, Unix `nohup`)

**Config Example (`config.yaml`):**
```yaml
no_tune: false
quick: false
no_charts: false
no_save: false
feature_select: true
```

**Usage:**
```powershell
# Interactive with full logging
python train_rf.py

# Background daemon with notification
python train_rf.py --daemon --notify

# Load settings from YAML
python train_rf.py --config config.yaml --daemon
```

### `vn30_simulation.py` — Monte-Carlo Engine
**Purpose:** Vectorized GBM simulation of VN30 equal-weight basket

**Features:**
- Per-stock drift (μ) & volatility (σ) from real log returns
- Vectorized GBM: (n_paths × n_stocks × n_days) in one go
- HOSE rules: ±7% daily clamp, 0.15% commission + 0.1% sell tax
- Risk metrics: VaR 95%, CVaR 95%, Sharpe, Max Drawdown per path
- 29 clean VN30 stocks from cache

**Usage:**
```powershell
python vn30_simulation.py --days 252 --paths 200 --seed 42
python vn30_simulation.py --days 504 --paths 1000 --no-chart
```

**Output Metrics:**
| Metric | Description |
|--------|-------------|
| Mean/Median Final | Portfolio value at horizon (start=100) |
| Best/Worst Path | Extreme outcomes |
| Annualized Return | Geometric annualization |
| Sharpe (mean path) | Risk-adjusted return |
| Mean Max Drawdown | Average peak-to-trough |
| Win Rate | % paths ending >100 |
| VaR 95% | 5th percentile final value |
| CVaR 95% | Mean of worst 5% paths |

### `vn30_visualize.py` — 8-Panel Dashboard
**Purpose:** Publication-quality multi-panel visualization

**Panels:**
1. **Fan Chart** — Index paths P5/P10/P25/P50/P75/P90/P95
2. **Median Equity Curve** — Buy & hold vs index
3. **Final Value Distribution** — Histogram + KDE + mean/median lines
4. **Risk Metrics Table** — Formatted summary (Win Rate, Sharpe, VaR, CVaR, etc.)
5. **Drawdown Fan Chart** — Percentile bands of drawdown % over time
6. **Annualized Return Distribution** — Histogram of annualized returns
7. **Rolling Sharpe** — 21-day rolling Sharpe of median path
8. **Equity Curve Heatmap** — Sample of 50 paths as color matrix

**Usage:**
```powershell
python vn30_visualize.py --days 252 --paths 500
python vn30_visualize.py --days 504 --paths 1000 --no-save
```

## Artifacts Produced

| File | Description |
|------|-------------|
| `vn30_ohlcv_cache.csv` | Cached OHLCV (35K rows, 30 symbols, 2021-2026) |
| `rf_model.joblib` | Trained RandomForestClassifier |
| `rf_scaler.joblib` | Fitted StandardScaler |
| `rf_meta.json` | Features, threshold, AUC, params, timestamp |
| `rf_predictions.csv` | Test set predictions for backtesting |
| `rf_results.png` | 2×2: Feature importance, ROC, PR, prob distribution |
| `vn30_sim_result.png` | Median path equity curve |
| `vn30_sim_visual.png` | 8-panel dashboard |

## Performance (Latest Run)

```
ROC-AUC:          0.6061  (baseline ~0.589, +1.7%)
PR-AUC (AP):      0.4042
F1-optimal threshold: 0.219
Test F1:          0.488
Today's Signals:  29 BUY / 1 HOLD (VCB)
Best CV AUC:      0.5981
Best params:      n_estimators=250, max_depth=4, max_features=log2, min_samples_leaf=10
```

```
VN30 Simulation (252 days, 200 paths):
Mean final:       115.2  |  Ann. Return: 15.2%
Sharpe:           2.20   |  Max DD: -3.9%
Win Rate:         99.0%  |  VaR 95%: 102.6
```

## Requirements

- Python ≥ 3.10
- `vnstock` ≥ 4.0.5 (requires API key from https://vnstocks.com/settings)
- `vnai` ≥ 2.5.6
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `pyyaml`

## Project Structure

```
├── stock_signal_random_forest.py   # Main ML pipeline
├── train_rf.py                     # Background trainer
├── vn30_simulation.py              # Monte-Carlo engine
├── vn30_visualize.py               # 8-panel dashboard
├── vn30_ohlcv_cache.csv            # Cached price data
├── rf_*.joblib / rf_*.json         # Model artifacts
├── rf_results.png                  # ML evaluation charts
├── vn30_sim_*.png                  # Simulation charts
└── train_log_*.txt                 # Training logs
```

## Tips

- **Cache-first design**: All scripts reuse `vn30_ohlcv_cache.csv` — no API calls after first run
- **Reproducibility**: Fixed `random_state=42` everywhere, explicit seeds for MC
- **No leakage**: Embargo gaps (3 days) between train/validation/test splits
- **Background runs**: Use `train_rf.py --daemon --notify` for overnight training
- **Config-driven**: Store flags in `config.yaml` for repeatable runs