#!/usr/bin/env python3
"""
Deep Reinforcement Learning for Stock Trading — same DQN recipe as Minesweeper,
applied to the VN30 stock predictor.

  State    -> last 10 days of relative features + current position
  Action   -> BUY / HOLD / SELL
  Reward   -> daily P&L (in %) minus transaction costs
              the agent earns money by correctly timing buys & sells
  Learning -> DQN with experience replay + target network + epsilon-greedy

Usage:
    python stock_rl.py                       # train on FPT, then backtest + chart
    python stock_rl.py --symbol HPG          # pick another symbol
    python stock_rl.py --symbols VN30        # train/backtest on all VN30
    python stock_rl.py --episodes 300        # train longer
"""

import os
import sys
import time
import random
import argparse
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from vnstock import Market, Listing

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

WINDOW = 10                     # days of history the agent "sees"
EPISODE_LEN = 60                # each episode = a random 60-day window
FEATURES = ["ret_1d", "ret_5d", "dist_sma20", "vol_ratio_20", "rsi_14"]
ACTIONS = ["BUY", "HOLD", "SELL"]
NUM_ACTIONS = len(ACTIONS)

TRANSACTION_COST = 0.2          # % per trade (buy or sell)
LEARN_EVERY = 4
BATCH_SIZE = 64
GAMMA = 0.95


# ─── DATA ────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbols, start, end):
    mrkt = Market()
    frames = []
    for sym in symbols:
        eq = mrkt.equity(symbol=sym)
        df = eq.ohlcv(start=start, end=end, interval="1D", count=2000)
        df = df.copy()
        df["symbol"] = sym
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "time"]).reset_index(drop=True)
    return out


def _rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def engineer_features(df):
    df = df.copy()
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)
    sma20 = df.groupby("symbol")["close"].transform(lambda x: x.rolling(20).mean())
    df["dist_sma20"] = (df["close"] / sma20) - 1.0
    vol20 = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())
    df["vol_ratio_20"] = df["volume"] / vol20
    df["rsi_14"] = df.groupby("symbol")["close"].transform(lambda x: _rsi(x))
    return df.dropna(subset=FEATURES).reset_index(drop=True)


# ─── THE TRADING ENVIRONMENT ─────────────────────────────────────────────────
class TradingEnv:
    """One episode = walking through a random 60-day window of one stock.

    Windows are sampled from the full history, so the agent sees many
    different market regimes while training.
    """

    def __init__(self, df, window=WINDOW, episode_len=EPISODE_LEN):
        self.features = df[FEATURES].values.astype(np.float32)
        self.prices = df["close"].values.astype(np.float64)
        self.window = window
        self.episode_len = episode_len
        self.total_steps = len(self.prices)
        self.max_start = self.total_steps - self.window - self.episode_len - 1

    def reset(self):
        self.start = np.random.randint(0, max(1, self.max_start))
        self.t = self.start + self.window
        self.end = self.start + self.window + self.episode_len
        self.position = 0          # 0 = flat, 1 = holding
        self.cash = 0.0            # cumulative P&L in %
        return self._state()

    def reset_full(self):
        """Reset to walk the ENTIRE history (used for backtesting)."""
        self.start = 0
        self.t = self.window
        self.end = self.total_steps - 1
        self.position = 0
        self.cash = 0.0
        return self._state()

    def _state(self):
        """Last W days of features + current position (0 or 1)."""
        feat = self.features[self.t - self.window:self.t].flatten()
        return np.concatenate([feat, [self.position]]).astype(np.float32)

    def step(self, action):
        prev_price = self.prices[self.t]
        next_price = self.prices[self.t + 1]
        day_return_pct = (next_price / prev_price - 1.0) * 100.0

        # 0 = BUY (go long), 1 = HOLD (do nothing), 2 = SELL (exit)
        if action == 0 and self.position == 0:      # open position
            self.position = 1
            trade_cost = TRANSACTION_COST
        elif action == 2 and self.position == 1:    # close position
            self.position = 0
            trade_cost = TRANSACTION_COST
        else:
            trade_cost = 0.0

        # Reward = P&L earned while holding, minus trading fees
        reward = day_return_pct * self.position - trade_cost

        self.cash += day_return_pct * self.position - trade_cost
        self.t += 1
        done = self.t >= self.end - 1

        return self._state(), reward, done


# ─── THE BRAIN ───────────────────────────────────────────────────────────────
def build_network(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_ACTIONS),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse")
    return model


class ReplayMemory:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, exp):
        self.memory.append(exp)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQNAgent:
    def __init__(self, input_dim):
        self.network = build_network(input_dim)
        self.target_network = build_network(input_dim)
        self.memory = ReplayMemory()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q = self.network.predict(state[None], verbose=0)[0]
        return int(np.argmax(q))

    def learn(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states = np.stack([b[0] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)

        future = self.target_network.predict(next_states, verbose=0).max(axis=1)
        targets = rewards + GAMMA * future * (1 - dones)

        q_all = self.network.predict(states, verbose=0)
        for i in range(BATCH_SIZE):
            q_all[i, actions[i]] = targets[i]

        self.network.fit(states, q_all, epochs=1, verbose=0)

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─── BACKTEST (no randomness, follow learned policy) ─────────────────────────
def backtest(agent, env, symbol=""):
    """Run one full history with the trained policy, record trades + equity."""
    agent.epsilon = 0.0
    state = env.reset_full()
    equity = [0.0]
    trades = []                  # (index, price, action)
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)[:3]
        equity.append(env.cash)
        if env.position == 1 and action == 0:
            trades.append((env.t, env.prices[env.t], "BUY"))
        elif env.position == 0 and action == 2:
            trades.append((env.t, env.prices[env.t], "SELL"))
        state = next_state

    return env, equity, trades


def plot_results(env, equity, trades, symbol, path):
    n = len(equity)
    dates = np.arange(n)
    prices = env.prices[env.window:env.window + n]

    # Buy-and-hold benchmark: buy first day, hold to end
    bh = (prices / prices[0] - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, equity, label="DQN Agent P&L %", linewidth=2, color="#1f77b4")
    ax.plot(dates, bh, label="Buy & Hold P&L %", linewidth=1.5,
            color="#d62728", alpha=0.8)
    for idx, price, act in trades:
        marker = "^" if act == "BUY" else "v"
        color = "green" if act == "BUY" else "red"
        pos = idx - env.window
        if 0 <= pos < n:
            ax.scatter(pos, equity[pos], marker=marker, color=color, s=60, zorder=5)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"DQN Trading Agent — {symbol} (2021–2025)")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio P&L (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {path}")


# ─── TRAIN ───────────────────────────────────────────────────────────────────
def train_agent(df, episodes, watch=False):
    env = TradingEnv(df)
    input_dim = WINDOW * len(FEATURES) + 1   # window of features + position
    agent = DQNAgent(input_dim)

    print("=" * 70)
    print(f"  Training DQN trading agent on {df['symbol'].iloc[0]} "
          f"({len(df)} days)")
    print("=" * 70)
    rewards_log = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.push((state, action, reward, next_state, done))
            if steps % LEARN_EVERY == 0:
                agent.learn()
            state = next_state
            total += reward
            steps += 1

        rewards_log.append(total)
        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.update_target()
            print(f"  Episode {ep:4d}/{episodes} | avg episode P&L: {np.mean(rewards_log[-10:]):+7.2f}%"
                  f" | explore: {agent.epsilon:.2f}")

    return agent, env


def main():
    parser = argparse.ArgumentParser(description="DQN stock trading agent")
    parser.add_argument("--symbol", type=str, default="FPT", help="stock symbol")
    parser.add_argument("--symbols", type=str, default=None,
                        help="'VN30' to train on the whole index")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--watch", action="store_true",
                        help="print trades live during training")
    args = parser.parse_args()

    # 1. Data
    print("[1/5] Fetching data ...")
    if args.symbols and args.symbols.upper() == "VN30":
        listing = Listing()
        syms = listing.symbols_by_group("VN30").tolist()
    else:
        syms = [args.symbol]
    raw = fetch_ohlcv(syms, START_DATE, END_DATE)
    df = engineer_features(raw)
    print(f"  {len(df)} rows, {df['symbol'].nunique()} symbols, "
          f"{raw['time'].min().date()} -> {raw['time'].max().date()}")

    # 2. Scale (no raw prices — only the relative features)
    print("[2/5] Scaling features ...")
    scaler = StandardScaler()
    df.loc[:, FEATURES] = scaler.fit_transform(df[FEATURES])

    # 3. Train
    print(f"[3/5] Training DQN ({args.episodes} episodes) ...")
    agent, env = train_agent(df, args.episodes, watch=args.watch)

    # 4. Backtest
    print("[4/5] Backtesting on full history ...")
    env2 = TradingEnv(df)
    _, equity, trades = backtest(agent, env2, syms[0])
    n_buys = sum(1 for _, _, a in trades if a == "BUY")
    n_sells = sum(1 for _, _, a in trades if a == "SELL")
    print(f"  Trades: {n_buys} buys, {n_sells} sells")

    # 5. Chart + summary
    print("[5/5] Generating chart ...")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "stock_rl_result.png")
    plot_results(env2, equity, trades, syms[0], out)

    final_pnl = equity[-1]
    bh = (env2.prices[-1] / env2.prices[env2.window] - 1.0) * 100.0
    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    print(f"  Agent final P&L     : {final_pnl:+.2f}%")
    print(f"  Buy & hold final P&L: {bh:+.2f}%")
    print(f"  {'AGENT BEAT THE MARKET' if final_pnl > bh else 'buy & hold won this time'} "
          f"(RL is hard — more episodes & tuning usually help)")


if __name__ == "__main__":
    main()
