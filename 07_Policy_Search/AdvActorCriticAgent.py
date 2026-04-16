import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from collections import deque

from plot_utils import plot_trainingsinformation


class ActorNetwork(nn.Module):
    """Outputs (mu, std) of a Gaussian policy for continuous action spaces."""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head  = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())
        self.std_head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Softplus())

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.std_head(h)


class CriticNetwork(nn.Module):
    """State-value function V(s)."""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class A2CAgent:
    """Advantage Actor-Critic (A2C) agent for continuous action spaces."""

    def __init__(self, env, config):
        self.env = env
        self.state_dim   = env.observation_space.shape[0]
        self.action_dim  = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        self.gamma      = config['DISCOUNT']
        self.lr_actor   = config['LEARNING_RATE']
        self.lr_critic  = config['LEARNING_RATE']
        self.hidden_dim = config['HIDDEN_DIM']
        self.batch_size = config['BATCH_SIZE']
        self.max_episodes  = config['EPISODES']
        self.max_timesteps = config['MAX_TIMESTEPS']
        self.logs = config['LOGS']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor  = ActorNetwork(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(self.device)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        self.reward_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []

    def _to_tensor(self, arr):
        # from_numpy avoids a data copy for contiguous float32 arrays
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self.device)

    def get_action(self, state):
        """Sample a clipped action from the Gaussian policy."""
        state_t = self._to_tensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            mu, std = self.actor(state_t)
        mu  = mu.squeeze(0).cpu().numpy()
        std = std.squeeze(0).cpu().numpy()
        action = np.random.normal(mu, std)
        return np.clip(action, -self.action_bound, self.action_bound)

    def update_batch(self, states, actions, rewards, next_states, dones):
        states_t      = self._to_tensor(states)
        next_states_t = self._to_tensor(next_states)
        actions_t     = self._to_tensor(actions)
        rewards_t     = self._to_tensor(rewards)
        dones_t       = self._to_tensor(dones)

        n = len(states_t)

        # Single critic forward pass over states + next_states concatenated
        # Halves the number of matrix multiplications compared to two separate calls
        self.critic.train()
        all_values  = self.critic(torch.cat([states_t, next_states_t], dim=0))
        values      = all_values[:n]
        next_values = all_values[n:].detach()  # no gradient needed for bootstrap targets

        targets   = rewards_t + (1 - dones_t) * self.gamma * next_values
        td_errors = (targets - values).detach()

        critic_loss = nn.functional.mse_loss(values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor.train()
        mu, std = self.actor(states_t)
        dist     = Normal(mu, std)
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        actor_loss = -(log_probs * td_errors).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self):
        """Train the A2C agent over the configured number of episodes."""
        scores_deque = deque(maxlen=100)

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            timestep = 0

            states, actions, rewards, next_states, dones = [], [], [], [], []

            for t in range(self.max_timesteps):
                timestep += 1
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(float(done))

                if len(states) == self.batch_size:
                    self.update_batch(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []

                state = next_state
                total_reward += reward
                if done:
                    break

            # Flush remaining transitions
            if states:
                self.update_batch(states, actions, rewards, next_states, dones)

            self.reward_episodes.append(total_reward)
            self.timesteps_per_episode.append(timestep)
            scores_deque.append(total_reward)
            self.average_score_100_episodes.append(np.mean(scores_deque))

            if total_reward == max(scores_deque):
                self.save_model(self.logs + f'/{total_reward:.2f}_agent')
            elif episode % 50 == 0:
                self.save_model(self.logs + f'/Episode_{episode}_agent')

            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Average(100): {np.mean(scores_deque):.2f}")

        self.save_data(self.logs, 'results.csv')

    def save_model(self, name):
        """Save actor and critic weights to <name>_actor.pth and <name>_critic.pth."""
        torch.save(self.actor.state_dict(),  name + '_actor.pth')
        torch.save(self.critic.state_dict(), name + '_critic.pth')

    def load_model(self, name):
        """Load actor and critic weights from .pth files."""
        self.actor.load_state_dict(torch.load(name + '_actor.pth',  map_location=self.device))
        self.critic.load_state_dict(torch.load(name + '_critic.pth', map_location=self.device))
        self.actor.eval()
        self.critic.eval()

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):
        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)
