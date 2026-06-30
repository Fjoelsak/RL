import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from collections import deque
from gymnasium import Env

from plot_utils import plot_trainingsinformation


def set_seed(seed):
    """Seed Python, NumPy and PyTorch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GaussianPolicyNetwork(nn.Module):
    """Policy network for continuous actions: a diagonal Gaussian policy.

    The network maps a state to the mean ``mu`` of the action distribution.
    The spread is a single state-independent ``log_std`` parameter (one per
    action dimension), which is more stable than a per-state std head that
    tends to collapse early.  The mean is left unsquashed; actions are clipped
    to the valid range only when sampling, because squashing ``mu`` with tanh
    saturates the head and kills its gradient near the action bounds.

    This is the continuous counterpart of the categorical ``PolicyNetwork``
    used for discrete action spaces: a softmax over logits becomes a Gaussian
    over a real-valued action.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        mu  = self.mu_head(self.body(x))
        std = self.log_std.exp().clamp(1e-3, 2.0)
        return mu, std


class ContinuousReinforceAgent:
    """REINFORCE with baseline (Monte-Carlo policy gradient) for continuous actions.

    Identical in spirit to the discrete ``ReinforceAgent``: roll out a full
    episode under the current stochastic policy, compute the discounted
    returns ``G_t``, and take one policy-gradient step weighted by those
    returns.  The only "baseline" here is the per-episode standardisation of
    the returns (zero mean, unit variance) — subtracting the mean acts as a
    variance-reducing baseline, dividing by the std keeps the gradient scale
    constant across episodes of different length.

    The single conceptual change versus the discrete agent is the policy
    distribution: a diagonal Gaussian (``Normal(mu, std)``) replaces the
    ``Categorical``.  The log-probability is summed over action dimensions.
    """

    def __init__(self, env, config):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Continuous (Box) action space, flat observation space.
        config : dict
            Keys: EPISODES, MAX_TIMESTEPS, DISCOUNT, LEARNING_RATE, HIDDEN_DIM,
            LOGS.  Optional: MAX_GRAD_NORM (default 0.5).
        """
        self.max_episodes  = config['EPISODES']
        self.max_timesteps = config['MAX_TIMESTEPS']
        self.gamma         = config['DISCOUNT']
        self.learning_rate = config['LEARNING_RATE']
        self.hidden_dim    = config['HIDDEN_DIM']
        self.max_grad_norm = config.get('MAX_GRAD_NORM', 0.5)
        self.input_dim     = env.observation_space.shape[0]
        self.output_dim    = env.action_space.shape[0]
        self.action_bound  = float(env.action_space.high[0])
        self.logs          = config['LOGS']

        # Seed everything first so the network init and the whole run are
        # reproducible. SEED=None leaves the RNGs untouched (non-deterministic).
        self.seed = config.get('SEED', None)
        if self.seed is not None:
            set_seed(self.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = GaussianPolicyNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Storage for one episode
        self.states  = []
        self.actions = []
        self.rewards = []

        self.reward_episodes            = []
        self.timesteps_per_episode      = []
        self.average_score_100_episodes = []

    def get_action(self, state):
        """Sample a continuous action from the current Gaussian policy.

        Draws ``a ~ Normal(mu(s), std)`` and clips it to the valid action
        range.  Exploration is built into the policy through the Gaussian
        noise, so no external exploration schedule is needed.
        """
        state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            mu, std = self.policy_net(state_t)
        mu  = mu.squeeze(0).cpu().numpy()
        std = std.cpu().numpy()
        action = np.random.normal(mu, std)
        return np.clip(action, -self.action_bound, self.action_bound)

    def store_transition(self, state, action, reward):
        """Append one (state, action, reward) tuple to the current episode buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def _compute_discounted_rewards(self):
        """Compute the standardised discounted returns G_t for the stored episode.

        Walks the reward list backwards to accumulate
        ``G_t = r_t + gamma * G_{t+1}``, then standardises the returns
        (zero mean, unit variance).  The mean acts as a variance-reducing
        baseline; the std keeps the gradient scale constant across episodes.
        """
        discounted = np.zeros(len(self.rewards), dtype=np.float32)
        cumulative = 0.0
        for t in reversed(range(len(self.rewards))):
            cumulative = self.rewards[t] + self.gamma * cumulative
            discounted[t] = cumulative
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-9)
        return discounted

    def _update_policy(self, episode: int):
        """Apply one REINFORCE gradient update from the finished episode.

        Recomputes the Gaussian policy for all visited states in a single
        batch, forms the policy-gradient loss from the summed log-probabilities
        weighted by the standardised returns, takes one (gradient-clipped)
        optimizer step, and clears the episode buffers.
        """
        discounted_rewards = self._compute_discounted_rewards()

        states_t  = torch.from_numpy(np.array(self.states, dtype=np.float32)).to(self.device)
        actions_t = torch.from_numpy(np.array(self.actions, dtype=np.float32)).to(self.device)
        returns_t = torch.from_numpy(discounted_rewards).to(self.device)

        self.policy_net.train()
        mu, std = self.policy_net(states_t)
        dist = Normal(mu, std.expand_as(mu))
        # Sum the per-dimension log-probs to get log pi(a_t | s_t).
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        # REINFORCE loss = negative of  sum_t log pi(a_t | s_t) * G_t.
        loss = -(log_probs * returns_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []

    def train(self, env: Env):
        """Train the policy network over the configured number of episodes.

        For each episode the agent rolls out a full trajectory under the current
        policy, stores every transition, and then performs one Monte-Carlo policy
        gradient update.  Per-episode metrics are written to ``results.csv`` at
        the end so the run can be plotted with :func:`plot_trainingsinformation`.
        """
        os.makedirs(self.logs, exist_ok=True)
        scores_deque = deque(maxlen=100)
        best_score   = -np.inf

        for episode in range(self.max_episodes):
            # Seed the environment's RNG on the very first reset for reproducibility.
            state, _ = env.reset(seed=self.seed if episode == 0 else None)
            total_reward = 0.0
            timestep_per_episode = 0

            for t in range(self.max_timesteps):
                timestep_per_episode += 1
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                self.store_transition(state, action, reward)
                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break

            self._update_policy(episode)

            scores_deque.append(total_reward)
            self.reward_episodes.append(total_reward)
            self.timesteps_per_episode.append(timestep_per_episode)
            avg = float(np.mean(scores_deque))
            self.average_score_100_episodes.append(avg)

            if avg > best_score:
                best_score = avg
                self.save_model(self.logs + '/best_agent')

            if episode % 50 == 0:
                print(f"Episode {episode}\tReward: {total_reward:.2f}\t"
                      f"Average Score: {avg:.2f}")

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
        """Write the per-episode training metrics to ``<logdir>/<name>`` as CSV."""
        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)
