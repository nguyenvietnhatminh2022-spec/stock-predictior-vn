#!/usr/bin/env python3
"""
Deep Reinforcement Learning from scratch — a DQN agent learns to navigate a grid.

Everything is built from scratch so you can see ALL the parts:
  1. Environment  — the grid world the agent lives in
  2. Neural Net   — the "brain" that predicts which actions are good
  3. Experience Replay — learns from past memories, not just the last step
  4. Target Network    — helps keep training stable
  5. Epsilon-greedy    — explores first, then exploits what it learned

The task: start at (0,0), reach the goal (5,5). Each step costs -1,
reaching the goal gives +10. The agent must LEARN a good path by trying.

Run it:  python rl_gridworld.py
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

GRID_SIZE = 6
START = (0, 0)
GOAL = (5, 5)
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # right, down, left, up
NUM_ACTIONS = len(ACTIONS)

# ─── STEP 1: THE ENVIRONMENT ─────────────────────────────────────────────────
class GridWorld:
    """A tiny grid the agent walks around in."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new episode: agent returns to (0,0)."""
        self.state = START
        return self._normalize(self.state)

    def step(self, action_idx):
        """Take an action, return (next_state, reward, done)."""
        row, col = self.state
        dr, dc = ACTIONS[action_idx]
        new_row, new_col = row + dr, col + dc

        # Wall bounce: can't leave the grid
        if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
            self.state = (new_row, new_col)

        # Reward logic
        if self.state == GOAL:
            return self._normalize(self.state), +10.0, True   # reached goal!
        return self._normalize(self.state), -1.0, False       # small cost each step

    def _normalize(self, state):
        """Turn (row, col) into 2 numbers between 0 and 1 for the network."""
        return np.array([state[0] / (GRID_SIZE - 1), state[1] / (GRID_SIZE - 1)],
                        dtype=np.float32)


# ─── STEP 2: THE BRAIN (neural network) ──────────────────────────────────────
def build_network():
    """Predicts a 'Q-value' for each action. Q = expected total future reward."""
    model = keras.Sequential([
        layers.Input(shape=(2,)),       # input: (row, col) position
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(NUM_ACTIONS),      # output: 4 values, one per action
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="mse")
    return model


# ─── STEP 3: EXPERIENCE REPLAY MEMORY ────────────────────────────────────────
class ReplayMemory:
    """Stores past (state, action, reward, next_state, done) memories."""
    def __init__(self, capacity=10000):
        self.memory = []
        self.capacity = capacity

    def push(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)          # forget the oldest memory

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))


# ─── STEP 4: THE DQN AGENT ───────────────────────────────────────────────────
class DQNAgent:
    def __init__(self):
        self.network = build_network()          # the main "brain"
        self.target_network = build_network()   # a stable copy, updated slowly
        self.memory = ReplayMemory()
        self.epsilon = 1.0         # exploration rate: 1.0 = act randomly
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995 # gets more confident each episode

    def choose_action(self, state):
        """Epsilon-greedy: explore randomly OR use the learned Q-values."""
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)   # explore
        q_values = self.network.predict(state[None, :], verbose=0)[0]
        return int(np.argmax(q_values))                  # exploit

    def learn(self, batch_size=64, gamma=0.9):
        """Update the network using a batch of past memories."""
        if len(self.memory.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            # Q_target = reward + gamma * (best future Q from the NEXT state)
            # gamma < 1 means "future rewards are worth slightly less"
            if done:
                target = reward
            else:
                future_q = np.max(self.target_network.predict(
                    next_state[None, :], verbose=0)[0])
                target = reward + gamma * future_q

            # We only update the Q-value of the action actually taken
            q_pred = self.network.predict(state[None, :], verbose=0)[0]
            q_pred[action] = target
            states.append(state)
            targets.append(q_pred)

        self.network.fit(np.array(states), np.array(targets),
                         epochs=1, verbose=0)

    def update_target_network(self):
        """Copy the main brain into the stable target brain."""
        self.target_network.set_weights(self.network.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─── MAIN TRAINING LOOP ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DQN training: navigating a 6x6 grid")
    print("=" * 60)

    env = GridWorld()
    agent = DQNAgent()

    NUM_EPISODES = 800
    TARGET_UPDATE_EVERY = 20     # refresh the stable brain every 20 episodes
    rewards_per_episode = []
    path_lengths = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Remember this experience for later learning
            agent.memory.push((state, action, reward, next_state, done))

            # Learn every few steps
            if steps % 4 == 0:
                agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        path_lengths.append(steps)

        agent.decay_epsilon()              # explore less over time
        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()  # stabilize training

        # Progress report every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            avg_steps = np.mean(path_lengths[-50:])
            print(f"  Episode {episode:4d} | avg reward: {avg_reward:6.1f} "
                  f"| avg path length: {avg_steps:5.1f} "
                  f"| explore rate: {agent.epsilon:.2f}")

    # ─── Show the learned policy ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Learned policy: best action at each grid cell")
    print("  (R=right D=down L=left U=up   *=goal)")
    print("=" * 60)
    arrows = {0: "R", 1: "D", 2: "L", 3: "U"}
    for row in range(GRID_SIZE):
        line = []
        for col in range(GRID_SIZE):
            if (row, col) == GOAL:
                line.append(" * ")
            else:
                state = np.array([row / (GRID_SIZE - 1), col / (GRID_SIZE - 1)],
                                 dtype=np.float32)
                q = agent.network.predict(state[None, :], verbose=0)[0]
                line.append(f" {arrows[int(np.argmax(q))]} ")
        print("   " + "|".join(line))

    # ─── Final verdict ───────────────────────────────────────────────────────
    last_avg = np.mean(rewards_per_episode[-100:])
    optimal_reward = 0 + (GRID_SIZE * 2 - 2) * (-1)   # shortest path reward
    print(f"\n  Shortest path length: {GRID_SIZE * 2 - 2} steps "
          f"(reward {optimal_reward})")
    print(f"  Final 100-episode avg reward: {last_avg:.1f}")
    if last_avg > -5:
        print("  SUCCESS: the agent learned to reach the goal quickly!")
    else:
        print("  The agent is still wandering. Try more episodes or a lower gamma.")


if __name__ == "__main__":
    main()