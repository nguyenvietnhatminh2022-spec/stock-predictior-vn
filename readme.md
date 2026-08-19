==============================================================
VN30 STOCK PREDICTION & SIMULATION — README
==============================================================

What this project does
----------------------
This project studies the Vietnamese stock market (VN30 index - the
30 largest, most liquid stocks on HOSE) using real historical data
fetched live from the vnstock library:

  1. Predict whether a stock will go up within 3 days (ML models).
  2. Simulate the VN30 basket forward in time with a Buy & Hold strategy.
  3. Visualize the results as PNG images or as pure-ASCII charts
     printed right in your terminal.

The simulation respects real Vietnamese market rules:
  - +-7% daily price limit (HOSE)
  - ~0.15% broker fee per side + 0.1% sell tax


==============================================================
SETUP (run once)
==============================================================

1. Open a terminal in this folder.

2. Create/use a Python virtual environment (Python >= 3.10):

       python -m venv ~/.venv
       ~/.venv\Scripts\activate          (Windows)
       source ~/.venv/bin/activate       (Mac / Linux)

3. Install the required packages:

       python -m pip install -U pip
       python -m pip install -U vnstock vnai numpy pandas scikit-learn
       python -m pip install matplotlib tensorflow   (for the RL demos + PNG charts)

   (vnaiv2 / vnstock needs an API key configured once — see
    https://vnstocks.com/settings and follow the vnstock docs.)


==============================================================
SCRIPTS — WHAT EACH ONE DOES
==============================================================

1) stock_signal_random_forest.py
   ------------------------------------------------
   ML signal predictor. Fetches all VN30 stocks, builds
   stationary features (ret_1d, ret_5d, dist_sma20, vol_ratio_20,
   rsi_14), labels "will price rise >1.5% in next 3 days?",
   splits chronologically with an embargo gap, then trains a
   Random Forest and reports ROC-AUC + classification report.

   Run:  python stock_signal_random_forest.py

2) model_comparison.py
   ------------------------------------------------
   Compares 5 classifiers (Logistic Regression, Random Forest,
   Gradient Boosting, SVM, KNN) on the same signal task.
   Saves PNG graphs (accuracy/AUC, scatter, summary table):
       model_summary_table.png
       model_scatter_accuracy_auc.png
       model_buy_signals.png
       model_comparison.png

   Run:  python model_comparison.py

3) stock_rl.py
   ------------------------------------------------
   Deep Reinforcement Learning (DQN) trading agent. Learns a
   BUY / HOLD / SELL policy on historical data (default FPT),
   then backtests the trained agent and saves a chart:
       stock_rl_result.png

   Run:  python stock_rl.py
   Options:  --symbol HPG        (pick a stock)
             --episodes 120      (training length)
             --watch             (print trades live)

4) vn30_simulation.py
   ------------------------------------------------
   THE MAIN SIMULATION. Uses real VN30 stock history to estimate
   each stock's drift (mu) and volatility (sigma), then runs a
   Monte-Carlo simulation (Geometric Brownian Motion, daily moves
   clamped to +-7%) of the equal-weight VN30 basket.

   On every simulated path it applies a Buy & Hold strategy
   (buy day 1, hold to the horizon, pays fees + tax).
   Prints a comparison table (mean/median/best/worst final value,
   annualized return, Sharpe, drawdown, win rate) and saves:
       vn30_sim_result.png
   First run downloads data into the cache file:
       vn30_ohlcv_cache.csv

   Run:  python vn30_simulation.py --days 252 --paths 200
   Options:  --days <n>   simulation length in trading days (default 252 = 1 year)
             --paths <n>  number of Monte-Carlo paths (default 200)
             --seed <n>   random seed for reproducibility (default 42)

5) vn30_visualize.py
   ------------------------------------------------
   Pretty 4-panel PNG chart of the same simulation
   (fan chart of index paths, median equity curve, final-value
   histogram, win-rate bar). Output:
       vn30_sim_visual.png

   Run:  python vn30_visualize.py --days 252 --paths 500

6) ascii_viz.py
   ------------------------------------------------
   Terminal-friendly version of the visualization - draws the
   median equity curve and final-value histogram using plain
   ASCII characters, no matplotlib needed.

   Run:  python ascii_viz.py --days 252 --paths 150
   Options:  --width 88 / --height 14   (chart size)

Bonus / learning demos (not part of the VN30 analysis):

7) rl_gridworld.py
   ------------------------------------------------
   Toy DQN demo - an agent learns to navigate a small grid from
   scratch (shows experience replay, target network, epsilon-greedy).

   Run:  python rl_gridworld.py

8) minesweeper_rl.py
   ------------------------------------------------
   DQN agent that learns to play Minesweeper (flags mines,
   reveals safe cells). Bigger/complex demo.

   Run:  python minesweeper_rl.py


==============================================================
QUICK START — RECOMMENDED WORKFLOW
==============================================================

    python vn30_simulation.py --days 252 --paths 200   # run the simulation
    python vn30_visualize.py   --days 252 --paths 500  # pretty PNG chart
    python ascii_viz.py        --days 252 --paths 150  # ASCII chart in terminal


==============================================================
KEY FILES PRODUCED
==============================================================

    vn30_ohlcv_cache.csv      cached VN30 price history (used by sim/viz)
    vn30_sim_result.png       simulation table chart (simulation script)
    vn30_sim_visual.png       ​4-panel visualization (visualize script)
    stock_rl_result.png       DQN agent backtest chart
    model_*.png               ML model comparison charts

==============================================================
TIP
==============================================================
The whole project works fine with just numpy + pandas + scikit-learn
plus vnstock. matplotlib and tensorflow are only needed for the PNG
charts and the RL demos. If you only want the ASCII simulation,
install vnstock + numpy + pandas and run ascii_viz.py.
