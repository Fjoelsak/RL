import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque
from gymnasium import Env

from plot_utils import plot_trainingsinformation


class PolicyNetwork(nn.Module):
    """Policy network outputting raw logits — softmax is handled by Categorical.

    Using logits instead of an explicit Softmax layer avoids a separate
    exp() + sum normalization step and is numerically more stable
    (Categorical applies log_softmax internally for log_prob).
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReinforceAgent:
    """REINFORCE (Monte-Carlo policy gradient) agent for discrete action spaces."""

    def __init__(self, env, config):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Discrete action space, flat observation space.
        config : dict
            Keys: EPISODES, MAX_TIMESTEPS, DISCOUNT, LEARNING_RATE, HIDDEN_DIM, LOGS.
        """
        self.max_episodes = config['EPISODES']
        self.max_timesteps = config['MAX_TIMESTEPS']
        self.gamma = config['DISCOUNT']
        self.learning_rate = config['LEARNING_RATE']
        self.hidden_dim = config['HIDDEN_DIM']
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.logs = config['LOGS']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = PolicyNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Storage for one episode
        self.states = []
        self.actions = []
        self.rewards = []

        self.reward_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []

    def get_action(self, state):
        """Sample an action from the policy. Returns (action, probability)."""
        state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_t).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        return action, dist.probs[action].item()

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def _compute_discounted_rewards(self):
        """Compute normalized discounted returns for the stored episode."""
        discounted = np.zeros(len(self.rewards), dtype=np.float32)
        cumulative = 0.0
        for t in reversed(range(len(self.rewards))):
            cumulative = self.rewards[t] + self.gamma * cumulative
            discounted[t] = cumulative
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-9)
        return discounted

    def _update_policy(self, episode: int):
        """Apply one REINFORCE gradient update and clear episode storage."""
        discounted_rewards = self._compute_discounted_rewards()

        states_t  = torch.from_numpy(np.array(self.states, dtype=np.float32)).to(self.device)
        actions_t = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        returns_t = torch.from_numpy(discounted_rewards).to(self.device)

        self.policy_net.train()
        logits = self.policy_net(states_t)
        # Categorical(logits=...) applies log_softmax internally — numerically stable
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        loss = -(log_probs * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []

    def train(self, env: Env):
        """Train the policy network over the configured number of episodes."""
        os.makedirs(self.logs, exist_ok=True)
        all_rewards = []
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.max_episodes):
            state, _ = env.reset()
            total_reward = 0
            timestep_per_episode = 0
            done = False

            for t in range(self.max_timesteps):
                timestep_per_episode += 1
                action, prob = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                self.store_transition(state, action, reward)
                state = next_state
                total_reward += reward

                done = terminated or truncated
                if done:
                    break

            self._update_policy(episode)
            all_rewards.append(total_reward)

            if done:
                if total_reward > max_reward:
                    self.save_model(self.logs + '/' + str(total_reward) + '_agent')
                elif episode % 50 == 0:
                    self.save_model(self.logs + '/Episode_' + str(episode) + '_agent')
                    print('Episode:\t', episode, '\t Average Score:\t', np.mean(scores_deque))

            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}")

            scores_deque.append(total_reward)
            max_reward = max(total_reward, max_reward)

            self.reward_episodes.append(total_reward)
            self.timesteps_per_episode.append(timestep_per_episode)
            self.average_score_100_episodes.append(np.mean(scores_deque))

        self.save_data(self.logs, 'results.csv')

    def save_model(self, name):
        """Save policy network weights to <name>.pth."""
        torch.save(self.policy_net.state_dict(), name + '.pth')

    def load_model(self, name):
        """Load policy network weights from <name>.pth."""
        self.policy_net.load_state_dict(torch.load(name + '.pth', map_location=self.device))
        self.policy_net.eval()

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):
        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)

    def test(self, env, name, TOTAL_EPISODES=10):
        """Load weights and run evaluation episodes."""
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
                action, _ = self.get_action(cur_state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done and episode_len > 475:
                    episodes_won += 1
                cur_state = next_state
                episode_reward += reward

            print('EPISODE_REWARD', episode_reward)

        print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')
