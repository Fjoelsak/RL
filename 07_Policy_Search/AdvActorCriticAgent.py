import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from collections import deque

from plot_utils import plot_trainingsinformation


def set_seed(seed):
    """Seed Python, NumPy and PyTorch for reproducible training runs.

    Note: on CPU, small non-determinism from floating-point reduction order
    can remain, but the overall trajectory of a run becomes reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActorCriticNet(nn.Module):
    """Shared backbone with separate actor and critic heads.

    Sharing features ensures that the critic's value-learning signal also
    improves the representation used by the actor.  A state-independent
    log_std parameter avoids the state-dependent std-collapse that can
    occur with a per-state std head.  The actor mean is left unsquashed;
    actions are clipped to the valid range only when sampling.  Squashing
    mu with tanh saturates the head and kills its gradient once mu drifts
    towards the action bounds, so the policy mean can no longer move.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound=1.0,
                 std_min=0.2, std_max=2.0):
        super().__init__()
        self.action_bound = action_bound
        # Lower- and upper-bound the policy std.  The lower bound (std_min) is
        # the important one: it slows down the variance-collapse spiral in
        # which a shrinking std inflates the log-probs, blows up the actor
        # gradient and drives the policy to a degenerate deterministic mode.
        self.log_std_min = float(np.log(std_min))
        self.log_std_max = float(np.log(std_max))
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mu_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std    = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h     = self.shared(x)
        mu    = self.mu_head(h)
        std   = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        value = self.value_head(h).squeeze(-1)
        return mu, std, value


class A2CAgent:
    """Advantage Actor-Critic (A2C) for continuous action spaces.

    The agent collects a short, fixed-length rollout (``n_steps`` transitions)
    by stepping continuously through the environment — crossing episode
    boundaries within a rollout — and performs **one** combined actor-critic
    update per rollout.  This is the key to making A2C work: updating every
    few steps yields thousands of updates per training run, whereas one
    update per finished episode (especially when early episodes last only a
    handful of steps) gives the critic far too little signal to learn from.

    Advantages are estimated with Generalized Advantage Estimation (GAE),
    which interpolates between low-variance TD(0) and high-variance Monte
    Carlo returns via ``gae_lambda``.  Exactly one gradient epoch is taken
    per rollout: re-using the same on-policy rollout for several epochs makes
    the policy drift off-policy and collapses training (that is PPO territory
    and requires importance-ratio clipping, which plain A2C does not have).

    Only true terminal states drop the bootstrap term.  An episode that ends
    because of a time limit (truncation) is *not* terminal: the underlying
    process continues, so its final state must still be bootstrapped via
    V(s_{t+1}).  Treating truncation as termination would inject a biased
    target on every time-limited episode.

    Combined loss: actor + vf_coef * critic - ent_coef * entropy.
    """

    def __init__(self, env, config):
        self.env          = env
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = float(env.action_space.high[0])

        # Seed everything first so the network init and the whole run are
        # reproducible. SEED=None leaves the RNGs untouched (non-deterministic).
        self.seed = config.get('SEED', None)
        if self.seed is not None:
            set_seed(self.seed)

        self.gamma         = config['DISCOUNT']
        self.gae_lambda    = config.get('GAE_LAMBDA',    0.95)
        self.lr            = config['LEARNING_RATE']
        self.vf_coef       = config.get('VF_COEF',       0.5)
        self.ent_coef      = config.get('ENTROPY_COEF',  0.001)
        self.max_grad_norm = config.get('MAX_GRAD_NORM', 0.5)
        self.std_min       = config.get('STD_MIN',       0.2)
        self.std_max       = config.get('STD_MAX',       2.0)
        self.hidden_dim    = config['HIDDEN_DIM']
        self.n_steps       = config.get('N_STEPS',       8)
        self.total_steps   = config['TOTAL_STEPS']
        self.logs          = config['LOGS']

        # Collapse auto-stop: once the 100-episode average has reached a
        # meaningful level (collapse_min_score) and then drops below
        # collapse_frac * best_score, training continues for a short grace
        # period (collapse_grace_steps) — so the collapse stays visible in the
        # diagnostics plots — and then stops. Set COLLAPSE_FRAC=None to disable.
        self.collapse_frac        = config.get('COLLAPSE_FRAC',       0.3)
        self.collapse_min_score   = config.get('COLLAPSE_MIN_SCORE',  200.0)
        self.collapse_grace_steps = config.get('COLLAPSE_GRACE_STEPS', 4000)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net       = ActorCriticNet(self.state_dim, self.hidden_dim, self.action_dim,
                                        self.action_bound, self.std_min, self.std_max).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.reward_episodes            = []
        self.timesteps_per_episode      = []
        self.average_score_100_episodes = []

        # Training diagnostics, one entry per gradient update.  Watching these
        # is the key lesson: a sudden actor-loss spike, a collapsing entropy /
        # policy std, or an exploding approximate KL all flag that the policy
        # is taking a destructive step.  Plain A2C has no trust region to
        # prevent this, which is exactly what PPO adds (see notebook 72).
        self.diag_steps      = []   # cumulative env steps at each update
        self.diag_entropy    = []   # mean policy entropy
        self.diag_std        = []   # mean policy std
        self.diag_approx_kl  = []   # approx. KL between pre- and post-update policy
        self.diag_actor_loss = []   # actor (policy-gradient) loss
        self.diag_value_loss = []   # critic (value) loss

    def _to_tensor(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self.device)

    def get_action(self, state):
        """Sample an action from the Gaussian policy."""
        state_t = self._to_tensor(state).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            mu, std, _ = self.net(state_t)
        mu  = mu.squeeze(0).cpu().numpy()
        std = std.cpu().numpy()
        action = np.random.normal(mu, std)
        return np.clip(action, -self.action_bound, self.action_bound)

    def _compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation for one rollout.

        Walks the rollout backwards accumulating
        ``A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`` with
        ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)``.
        The bootstrap is dropped only on true terminal states (``dones``),
        never on time-limit truncation.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            The advantages and the corresponding value targets (returns).
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, next_states, dones):
        """One combined actor-critic update on a single n-step rollout.

        Computes GAE advantages and value targets from the rollout, then takes
        a single gradient step on the combined actor + critic + entropy loss.

        Returns
        -------
        dict
            Training diagnostics for this update: ``entropy``, ``std``,
            ``approx_kl`` (between the policy before and after the step),
            ``actor_loss`` and ``value_loss``.
        """
        states_t      = self._to_tensor(np.array(states))
        actions_t     = self._to_tensor(np.array(actions))
        next_states_t = self._to_tensor(np.array(next_states))
        rewards_a     = np.asarray(rewards, dtype=np.float32)
        dones_a       = np.asarray(dones,   dtype=np.float32)

        self.net.train()
        mu, std, values = self.net(states_t)

        with torch.no_grad():
            _, _, next_values = self.net(next_states_t)

        advantages, returns = self._compute_gae(
            rewards_a, values.detach().cpu().numpy(), next_values.cpu().numpy(), dones_a)
        advantages_t = self._to_tensor(advantages)
        returns_t    = self._to_tensor(returns)

        dist      = Normal(mu, std.expand_as(mu))
        log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)

        actor_loss  = -(log_probs * advantages_t).mean()
        critic_loss = nn.functional.mse_loss(values, returns_t)
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy.mean()

        # Snapshot the log-probs of the taken actions *before* the step so we
        # can measure how far the policy moved (approximate KL divergence).
        old_log_probs = log_probs.detach()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            new_mu, new_std, _ = self.net(states_t)
            new_dist = Normal(new_mu, new_std.expand_as(new_mu))
            new_log_probs = new_dist.log_prob(actions_t).sum(dim=-1)
            # Schulman's low-variance approximate KL estimator:
            # E[(r - 1) - log r] with r = pi_new / pi_old.
            log_ratio = new_log_probs - old_log_probs
            approx_kl = float((log_ratio.exp() - 1.0 - log_ratio).mean())

        return {
            'entropy':    float(entropy.mean()),
            'std':        float(std.mean()),
            'approx_kl':  approx_kl,
            'actor_loss': float(actor_loss),
            'value_loss': float(critic_loss),
        }

    def train(self):
        """Train the A2C agent for ``total_steps`` environment steps.

        Steps through the environment continuously, performing one update
        every ``n_steps`` transitions.  Episode boundaries are handled inside
        the rollout: when an episode ends the environment is reset and the
        return is logged, but the rollout keeps filling until ``n_steps`` is
        reached.  Metrics are recorded per finished episode so the run can be
        plotted with :func:`plot_trainingsinformation`.

        If collapse auto-stop is enabled (``COLLAPSE_FRAC`` not None), training
        ends a short grace period after the 100-episode average drops far below
        its best value — keeping the run reproducible-ish in length regardless
        of when the (seed-dependent) collapse happens.
        """
        os.makedirs(self.logs, exist_ok=True)
        scores_deque = deque(maxlen=100)
        best_score   = -np.inf

        # Seed the environment's RNG on the first reset for reproducibility.
        state, _ = self.env.reset(seed=self.seed)
        episode_reward    = 0.0
        episode_timesteps = 0
        episode           = 0

        collapse_step = None   # step at which a collapse was first detected

        steps_done = 0
        while steps_done < self.total_steps:
            states, actions, rewards, next_states, dones = [], [], [], [], []

            for _ in range(self.n_steps):
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                # Only true termination drops the bootstrap; truncation
                # (time limit) does not, so store ``terminated`` here, not
                # ``terminated or truncated``.
                dones.append(float(terminated))

                state = next_state
                episode_reward    += reward
                episode_timesteps += 1
                steps_done        += 1

                if terminated or truncated:
                    scores_deque.append(episode_reward)
                    self.reward_episodes.append(episode_reward)
                    self.timesteps_per_episode.append(episode_timesteps)
                    avg = float(np.mean(scores_deque))
                    self.average_score_100_episodes.append(avg)

                    if avg > best_score:
                        best_score = avg
                        self.save_model(self.logs + '/best_agent')
                    if episode % 50 == 0:
                        print(f"Episode {episode}, Steps: {steps_done}, "
                              f"Reward: {episode_reward:.2f}, Average(100): {avg:.2f}")

                    # Detect a collapse: a steep drop of the moving average
                    # after it had reached a meaningful level.
                    if (self.collapse_frac is not None and collapse_step is None
                            and best_score >= self.collapse_min_score
                            and avg < self.collapse_frac * best_score):
                        collapse_step = steps_done
                        print(f"  -> collapse detected at step {steps_done} "
                              f"(avg {avg:.1f} < {self.collapse_frac:.0%} of best {best_score:.1f}); "
                              f"training {self.collapse_grace_steps} more steps then stopping.")

                    episode += 1
                    episode_reward    = 0.0
                    episode_timesteps = 0
                    state, _ = self.env.reset()

            diag = self.update(states, actions, rewards, next_states, dones)
            self.diag_steps.append(steps_done)
            self.diag_entropy.append(diag['entropy'])
            self.diag_std.append(diag['std'])
            self.diag_approx_kl.append(diag['approx_kl'])
            self.diag_actor_loss.append(diag['actor_loss'])
            self.diag_value_loss.append(diag['value_loss'])

            # Stop once the post-collapse grace period has elapsed.
            if collapse_step is not None and steps_done - collapse_step >= self.collapse_grace_steps:
                print(f"Stopping after collapse grace period at step {steps_done}.")
                break

        self.save_data(self.logs, 'results.csv')

    def get_diagnostics(self):
        """Return per-update training diagnostics as a DataFrame.

        Columns: ``Steps``, ``Entropy``, ``Std``, ``ApproxKL``, ``ActorLoss``,
        ``ValueLoss`` — one row per gradient update.  Use these to watch for
        the warning signs of an unstable policy: collapsing entropy / std,
        spiking approximate KL, or an exploding actor loss.
        """
        return pd.DataFrame({
            'Steps':     self.diag_steps,
            'Entropy':   self.diag_entropy,
            'Std':       self.diag_std,
            'ApproxKL':  self.diag_approx_kl,
            'ActorLoss': self.diag_actor_loss,
            'ValueLoss': self.diag_value_loss,
        })

    def save_model(self, name):
        torch.save(self.net.state_dict(), name + '.pth')

    def load_model(self, name):
        self.net.load_state_dict(torch.load(name + '.pth', map_location=self.device))
        self.net.eval()

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):
        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)
