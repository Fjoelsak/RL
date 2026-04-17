
from collections import deque
import os
import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from plot_utils import plot_trainingsinformation


class ReplayBuffer:
    """Circular buffer with pre-allocated numpy arrays.

    Avoids the per-step overhead of a deque of tuples:
    - O(1) insertion vs O(1) deque append (same), but no Python object per transition
    - O(batch) sampling via integer indexing instead of random.sample + zip
    - np.array() on a pre-allocated slice is zero-copy
    """

    def __init__(self, capacity, obs_dim):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        obs_shape = (capacity, *obs_dim)
        self.states      = np.zeros(obs_shape, dtype=np.float32)
        self.next_states = np.zeros(obs_shape, dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self._ptr]      = state
        self.next_states[self._ptr] = next_state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self._size


class DQNNetwork(nn.Module):
    """DQN: two hidden layers (24, 48, ReLU), linear output over action_dim."""

    def __init__(self, observation_dim, action_dim):
        super().__init__()
        input_size = int(np.prod(observation_dim))
        self.net = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay and a separate target network."""

    def __init__(self, env, config):
        self.episodes = config['EPISODES']
        self.epsilon = config['EPSILON']
        self.epsDecay = config['EPSILON_DECAY']
        self.minEps = config['MINIMUM_EPSILON']
        self.discount = config['DISCOUNT']
        self.miniBatchSize = config['MINIBATCH_SIZE']
        self.minReplayMem = config['MINIMUM_REPLAY_MEMORY']
        self.trainFrequency = config['TRAIN_FREQUENCY']
        self.updateTQNW = config['UPDATE_TARGETNW_STEPS']
        self.learningRate = config['LEARNING_RATE']
        self.visualization = config['VISUALIZATION']
        self.logs = config['LOGS']

        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape

        self.replay_memory = ReplayBuffer(config['REPLAY_MEMORY_SIZE'], self.observation_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DQNNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.targetmodel = DQNNetwork(self.observation_dim, self.action_dim).to(self.device)
        self.targetmodel.load_state_dict(self.model.state_dict())
        self.targetmodel.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.loss_fn = nn.MSELoss()

        self.counterDQNTrained = 0

        self.reward_episodes = []
        self.epsilon_over_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []

    def load_model(self, name):
        """Load model weights from a .pth file."""
        self.model.load_state_dict(torch.load(name + '.pth', map_location=self.device))
        self.model.eval()

    def save_model(self, name):
        """Save model weights to a .pth file."""
        torch.save(self.model.state_dict(), name + '.pth')

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.add(state, action, reward, next_state, done)

    def _to_tensor(self, arr):
        # from_numpy avoids a data copy for contiguous float32 arrays
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self.device)

    def trainDQN(self):
        self.counterDQNTrained += 1

        # ReplayBuffer.sample returns pre-allocated numpy slices — no zip/list needed
        cur_states, actions, rewards, next_states, dones = self.replay_memory.sample(self.miniBatchSize)

        cur_states_t  = self._to_tensor(cur_states)
        next_states_t = self._to_tensor(next_states)
        rewards_t     = self._to_tensor(rewards)
        dones_t       = self._to_tensor(dones)
        actions_t     = torch.from_numpy(actions).to(self.device)

        # Current Q-values for taken actions
        self.model.train()
        q_values = self.model(cur_states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values from frozen target network
        with torch.no_grad():
            next_q  = self.targetmodel(next_states_t).max(1).values
            targets = rewards_t + (1 - dones_t) * self.discount * next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.counterDQNTrained % self.updateTQNW == 0:
            self.targetmodel.load_state_dict(self.model.state_dict())

    def _predict_action(self, state):
        """Greedy action selection (no gradient tracking needed)."""
        self.model.eval()
        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def train(self, env):
        """Train the agent for the configured number of episodes."""
        logdir = self.logs
        os.makedirs(logdir, exist_ok=True)
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.episodes):
            cur_state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                episode_length += 1
                if self.visualization:
                    env.render()

                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = self._predict_action(cur_state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if done:
                    if episode_reward > max_reward:
                        self.save_model(logdir + '/' + str(episode_reward) + '_agent')
                    elif episode % 50 == 0:
                        self.save_model(logdir + '/Episode_' + str(episode) + '_agent')
                        print('Episode:\t', episode, '\t Average Score:\t', np.mean(scores_deque))

                self.memorize(cur_state, action, reward, next_state, done)
                cur_state = next_state

                if len(self.replay_memory) < self.minReplayMem:
                    continue

                if episode_length % self.trainFrequency == 0:
                    self.trainDQN()

            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Average(100): {np.mean(scores_deque):.2f}")

            if self.epsilon > self.minEps and len(self.replay_memory) > self.minReplayMem:
                self.epsilon *= self.epsDecay

            scores_deque.append(episode_reward)
            max_reward = max(episode_reward, max_reward)

            self.reward_episodes.append(episode_reward)
            self.epsilon_over_episodes.append(self.epsilon)
            self.timesteps_per_episode.append(episode_length)
            self.average_score_100_episodes.append(np.mean(scores_deque))

        self.save_data(logdir, 'results.csv')

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_epsilon='Epsilon over episodes',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):

        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_epsilon: self.epsilon_over_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)

    def test(self, env, name, TOTAL_EPISODES=10):
        """Load model weights and run evaluation episodes."""
        self.load_model(name)
        episodes_won = 0

        for _ in range(TOTAL_EPISODES):
            episode_reward = 0
            cur_state, _ = env.reset()
            done = False
            episode_len = 0

            while not done:
                env.render()
                episode_len += 1
                action = self._predict_action(cur_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done and episode_len > 475:
                    episodes_won += 1
                cur_state = next_state
                episode_reward += reward

            print('EPISODE_REWARD', episode_reward)

        print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')


class NaiveDQNAgent:
    """DQN without experience replay and without a target network.

    Each transition triggers an immediate online update using the main
    network for both action selection and bootstrap targets. Intended as
    an ablation baseline to demonstrate why both components are necessary.
    """

    def __init__(self, env, config):
        self.episodes     = config['EPISODES']
        self.epsilon      = config['EPSILON']
        self.epsDecay     = config['EPSILON_DECAY']
        self.minEps       = config['MINIMUM_EPSILON']
        self.discount     = config['DISCOUNT']
        self.learningRate = config['LEARNING_RATE']
        self.logs         = config['LOGS']

        self.action_dim = env.action_space.n
        self.obs_dim    = env.observation_space.shape

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = DQNNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.loss_fn   = nn.MSELoss()

        self.reward_episodes            = []
        self.epsilon_over_episodes      = []
        self.timesteps_per_episode      = []
        self.average_score_100_episodes = []

    def _to_tensor(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self.device)

    def _predict_action(self, state):
        self.model.eval()
        with torch.no_grad():
            q = self.model(self._to_tensor(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def _train_step(self, state, action, reward, next_state, done):
        """Online update on a single transition."""
        s  = self._to_tensor(state).unsqueeze(0)
        ns = self._to_tensor(next_state).unsqueeze(0)

        self.model.train()
        q_val = self.model(s)[0, action]

        with torch.no_grad():
            next_q = self.model(ns).max(1).values[0]
        target = reward + (1 - float(done)) * self.discount * next_q

        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env):
        """Train the naive agent for the configured number of episodes."""
        os.makedirs(self.logs, exist_ok=True)
        scores_deque = deque(maxlen=100)

        for episode in range(self.episodes):
            state, _ = env.reset()
            done           = False
            episode_reward = 0
            episode_length = 0

            while not done:
                episode_length += 1
                if np.random.uniform(0, 1) < self.epsilon:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = self._predict_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                self._train_step(state, action, reward, next_state, done)
                state = next_state

            if self.epsilon > self.minEps:
                self.epsilon *= self.epsDecay

            scores_deque.append(episode_reward)
            self.reward_episodes.append(episode_reward)
            self.epsilon_over_episodes.append(self.epsilon)
            self.timesteps_per_episode.append(episode_length)
            self.average_score_100_episodes.append(np.mean(scores_deque))

            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Average(100): {np.mean(scores_deque):.2f}")
