#!/usr/bin/env python3
"""
Minesweeper 100x100 with a Deep Reinforcement Learning (DQN) agent.

The agent learns by experience:
  - Flagging a mine correctly   -> reward  (+5)   "detects mine correctly"
  - Flagging a safe cell        -> penalty (-2)
  - Revealing a safe cell       -> reward  (+1 per cell opened)
  - Revealing a mine (stepping) -> big penalty (-10), episode ends
  - Clearing the whole board    -> bonus   (+30)

Architecture: a CNN looks at the whole 100x100 board and outputs
two Q-maps (one for "reveal", one for "flag"). Standard DQN with
experience replay + a target network.

Run it:
    python minesweeper_rl.py                # train then watch a demo game
    python minesweeper_rl.py --episodes 200 # train longer
    python minesweeper_rl.py --watch 5      # only watch 5 games (no training)
"""

import os
import sys
import time
import random
import argparse
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # quiet TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

GRID = 50                      # 50 x 50 board
MINE_DENSITY = 0.15            # 15% of cells are mines
NUM_MINES = int(GRID * GRID * MINE_DENSITY)

REWARD_FLAG_CORRECT = +5.0     # correctly detected a mine
REWARD_FLAG_WRONG = -2.0       # flagged a safe cell
REWARD_REVEAL_SAFE = +1.0      # per safe cell revealed (incl. flood fill)
REWARD_STEP_MINE = -10.0       # revealed a mine -> penalty
REWARD_WIN = +30.0             # cleared the whole board

MAX_STEPS = 250                # cap steps per episode
WATCH_EVERY_EPISODE = True     # render every episode live in the terminal
WATCH_DELAY = 0.05             # seconds between rendered steps


# ─── THE ENVIRONMENT ─────────────────────────────────────────────────────────
class Minesweeper:
    """The game world the agent interacts with."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Place mines and hide the board."""
        flat = np.zeros(GRID * GRID, dtype=bool)
        flat[:NUM_MINES] = True
        np.random.shuffle(flat)
        self.mines = flat.reshape(GRID, GRID)
        self.numbers = self._compute_numbers()
        self.revealed = set()              # revealed safe cells
        self.flags = set()                 # flagged cells
        self.mine_hit = None               # where the agent stepped on a mine
        return self._state(), self._valid_mask()

    def _compute_numbers(self):
        nums = np.zeros((GRID, GRID), dtype=np.int8)
        for r in range(GRID):
            for c in range(GRID):
                if not self.mines[r, c]:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < GRID and 0 <= nc < GRID
                                    and self.mines[nr, nc]):
                                nums[r, c] += 1
        return nums

    def _state(self):
        """Encode the board as 3 channels for the CNN."""
        c_number = np.zeros((GRID, GRID), dtype=np.float32)   # revealed nums /8
        c_hidden = np.ones((GRID, GRID), dtype=np.float32)    # 1 = not revealed
        c_flag = np.zeros((GRID, GRID), dtype=np.float32)     # 1 = flagged
        for (r, c) in self.revealed:
            c_number[r, c] = self.numbers[r, c] / 8.0
            c_hidden[r, c] = 0.0
        for (r, c) in self.flags:
            c_flag[r, c] = 1.0
        return np.stack([c_number, c_hidden, c_flag], axis=-1)

    def _valid_mask(self):
        """Cells that can still be acted on: hidden and not flagged."""
        mask = np.ones((GRID, GRID), dtype=bool)
        for (r, c) in self.revealed:
            mask[r, c] = False
        for (r, c) in self.flags:
            mask[r, c] = False
        return mask

    def step(self, action_type, r, c):
        """action_type: 0 = reveal, 1 = flag."""
        if action_type == 1:                       # ── FLAG ──
            self.flags.add((r, c))
            if self.mines[r, c]:
                return self._state(), self._valid_mask(), REWARD_FLAG_CORRECT, False, "flag_correct"
            return self._state(), self._valid_mask(), REWARD_FLAG_WRONG, False, "flag_wrong"

        # ── REVEAL ──
        if self.mines[r, c]:                       # stepped on a mine
            self.mine_hit = (r, c)
            self.revealed.add((r, c))
            return self._state(), self._valid_mask(), REWARD_STEP_MINE, True, "mine"

        opened = self._flood_reveal(r, c)          # opens 0-cells too
        reward = opened * REWARD_REVEAL_SAFE
        if len(self.revealed) == GRID * GRID - NUM_MINES:   # won
            return self._state(), self._valid_mask(), reward + REWARD_WIN, True, "win"
        return self._state(), self._valid_mask(), reward, False, "reveal"

    def _flood_reveal(self, r, c):
        """Reveal the cell and auto-open all connected 0-cells (like real Minesweeper)."""
        q = deque([(r, c)])
        opened = 0
        while q:
            rr, cc = q.popleft()
            if (rr, cc) in self.revealed or self.mines[rr, cc]:
                continue
            self.revealed.add((rr, cc))
            opened += 1
            if self.numbers[rr, cc] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < GRID and 0 <= nc < GRID
                                and (nr, nc) not in self.revealed
                                and (nr, nc) not in self.flags):
                            q.append((nr, nc))
        return opened

    def render(self, show_mines=False):
        """Draw the board in the terminal."""
        os.system("cls" if os.name == "nt" else "clear")
        lines = ["  " + "".join(f"{c % 10}" for c in range(GRID))]
        for r in range(GRID):
            row = f"{r % 10} "
            for c in range(GRID):
                if (r, c) == self.mine_hit:
                    row += "X"
                elif (r, c) in self.revealed:
                    v = self.numbers[r, c]
                    row += " " if v == 0 else str(v)
                elif (r, c) in self.flags:
                    row += "F"
                elif show_mines and self.mines[r, c]:
                    row += "*"
                else:
                    row += "."
            lines.append(row)
        print("\n".join(lines))
        flags_right = sum(1 for (r, c) in self.flags if self.mines[r, c])
        print(f"\n  Mines: {NUM_MINES}  Flagged: {len(self.flags)} "
              f"(correct: {flags_right})  Revealed: {len(self.revealed)}")


# ─── THE BRAIN (CNN) ─────────────────────────────────────────────────────────
def build_network():
    """Input: 100x100x3 board. Output: Q-reveal map and Q-flag map."""
    inp = layers.Input(shape=(GRID, GRID, 3))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    q_reveal = layers.Conv2D(1, 1, padding="same")(x)
    q_flag = layers.Conv2D(1, 1, padding="same")(x)
    model = keras.Model(inp, [q_reveal, q_flag])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss="mse")
    return model


# ─── DQN AGENT ───────────────────────────────────────────────────────────────
class ReplayMemory:
    def __init__(self, capacity=20000):
        self.memory = deque(maxlen=capacity)

    def push(self, exp):
        self.memory.append(exp)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQNAgent:
    def __init__(self):
        self.network = build_network()
        self.target_network = build_network()
        self.memory = ReplayMemory()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98

    def choose_action(self, state, valid_mask):
        """Epsilon-greedy over all valid (flag/reveal, cell) pairs."""
        cells = np.argwhere(valid_mask)
        if len(cells) == 0:
            return None

        if np.random.rand() < self.epsilon:            # EXPLORE: random valid move
            r, c = cells[np.random.randint(len(cells))]
            return (np.random.randint(2), int(r), int(c))

        qr, qf = self.network.predict(state[None], verbose=0)     # EXPLOIT
        qr = qr[0, ..., 0].copy()
        qf = qf[0, ..., 0].copy()
        qr[~valid_mask] = -1e9
        qf[~valid_mask] = -1e9
        if qr.max() >= qf.max():
            r, c = np.unravel_index(np.argmax(qr), qr.shape)
            return (0, int(r), int(c))
        r, c = np.unravel_index(np.argmax(qf), qf.shape)
        return (1, int(r), int(c))

    def learn(self, batch_size=16, gamma=0.9):
        if len(self.memory.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        states = np.stack([b[0] for b in batch])
        next_states = np.stack([b[3] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        rows = np.array([b[1][1] for b in batch], dtype=np.int32)
        cols = np.array([b[1][2] for b in batch], dtype=np.int32)
        types = np.array([b[1][0] for b in batch], dtype=np.int32)

        # Best future value from the stable target network (batched)
        qr_t, qf_t = self.target_network.predict(next_states, verbose=0)
        future = np.maximum(qr_t[..., 0], qf_t[..., 0]).max(axis=(1, 2))
        targets = rewards + gamma * future * (1 - dones)

        qr, qf = self.network.predict(states, verbose=0)
        qr = qr[..., 0]
        qf = qf[..., 0]
        for i in range(batch_size):                     # update only the taken action
            if types[i] == 0:
                qr[i, rows[i], cols[i]] = targets[i]
            else:
                qf[i, rows[i], cols[i]] = targets[i]

        self.network.fit(states, [qr[..., None], qf[..., None]],
                         epochs=1, verbose=0)

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─── TRAINING ────────────────────────────────────────────────────────────────
def train_agent(episodes, learn_every=8):
    env = Minesweeper()
    agent = DQNAgent()

    stats = {"flag_correct": 0, "flag_wrong": 0, "mines_hit": 0, "wins": 0}
    rewards_log = []

    print("=" * 64)
    print(f"  Training DQN on {GRID}x{GRID} Minesweeper "
          f"({NUM_MINES} mines)")
    print("=" * 64)

    for ep in range(1, episodes + 1):
        state, valid = env.reset()
        total = 0.0
        steps = 0
        done = False

        if WATCH_EVERY_EPISODE:
            env.render()
            print(f"\n  >>> Episode {ep}/{episodes} starting "
                  f"(explore rate {agent.epsilon:.2f})")

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state, valid)
            if action is None:
                break
            a_type, r, c = action
            next_state, next_valid, reward, done, kind = env.step(a_type, r, c)

            if WATCH_EVERY_EPISODE:
                label = "FLAG" if a_type == 1 else "REVEAL"
                print(f"  [step {steps}] {label} ({r}, {c}) -> reward {reward:+.1f} "
                      f"({kind})")
                env.render()
                time.sleep(WATCH_DELAY)

            agent.memory.push((state, (a_type, r, c), reward, next_state, done))
            if steps % learn_every == 0:
                agent.learn()

            if kind == "flag_correct":
                stats["flag_correct"] += 1
            elif kind == "flag_wrong":
                stats["flag_wrong"] += 1
            elif kind == "mine":
                stats["mines_hit"] += 1
            elif kind == "win":
                stats["wins"] += 1

            state, valid = next_state, next_valid
            total += reward
            steps += 1

        rewards_log.append(total)
        agent.decay_epsilon()
        if ep % 5 == 0:
            agent.update_target()
            avg = np.mean(rewards_log[-20:])
            print(f"  Episode {ep:4d}/{episodes} | avg reward (last 20): {avg:7.1f}"
                  f" | explore: {agent.epsilon:.2f}")

    print("\n  === TRAINING SUMMARY ===")
    print(f"  Mines correctly flagged : {stats['flag_correct']}   (+5 each)")
    print(f"  Safe cells wrongly flagged: {stats['flag_wrong']}   (-2 each)")
    print(f"  Mines stepped on        : {stats['mines_hit']}   (-10 each)")
    print(f"  Boards cleared          : {stats['wins']}   (+30 each)")
    last_avg = np.mean(rewards_log[-20:]) if rewards_log else 0
    print(f"  Final avg episode reward: {last_avg:.1f}")

    return agent, env, stats, last_avg


# ─── WATCH A GAME ────────────────────────────────────────────────────────────
def watch_game(agent, delay=0.3):
    """Play one game in the terminal with the trained agent (exploit only)."""
    env = Minesweeper()
    agent.epsilon = 0.0                       # no randomness
    state, valid = env.reset()
    total = 0.0
    done = False
    steps = 0
    while not done and steps < MAX_STEPS:
        env.render()
        print(f"  Step {steps}  Total reward: {total:.1f}")
        time.sleep(delay)
        action = agent.choose_action(state, valid)
        if action is None:
            break
        a_type, r, c = action
        label = "FLAG" if a_type == 1 else "REVEAL"
        print(f"  Agent {label}s cell ({r}, {c})")
        time.sleep(delay)
        next_state, next_valid, reward, done, kind = env.step(a_type, r, c)
        state, valid = next_state, next_valid
        total += reward
        steps += 1

    env.render(show_mines=True)               # reveal all mines at the end
    print(f"\n  Game over after {steps} moves. Total reward: {total:.1f} "
          f"({'BOARD CLEARED!' if done and len(env.revealed) == GRID*GRID - NUM_MINES else 'stepped on a mine'})")


def main():
    parser = argparse.ArgumentParser(description="Minesweeper DQN")
    parser.add_argument("--episodes", type=int, default=60,
                        help="training episodes (default 60)")
    parser.add_argument("--watch", type=int, default=1,
                        help="games to watch after training")
    parser.add_argument("--watch-only", type=int, default=0,
                        help="skip training and just watch a random agent")
    args = parser.parse_args()

    if args.watch_only:
        watch_game(DQNAgent(), delay=0.15)     # untrained -> random moves
    else:
        agent, _, _, last_avg = train_agent(args.episodes)
        print("\n" + "=" * 64)
        print("  Demo: watching the trained agent play")
        print("=" * 64)
        for _ in range(max(1, args.watch)):
            watch_game(agent, delay=0.25)


if __name__ == "__main__":
    main()
