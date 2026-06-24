import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from TileCoding import TileCoder

class LinearSarsaAgent:
    """
    A SARSA agent using linear function approximation over tile-coded features.

    The agent learns an action-value function Q(s, a) as a linear combination of
    binary tile-coding features (see :class:`TileCoder`). Because tile coding yields
    a sparse one-hot-per-tiling representation, the linear model reduces to summing
    the weights of the active tiles, and each on-policy SARSA update only touches
    those active weights.

    Action selection follows an ε-greedy policy, with ε decayed over the course of
    training to shift from exploration toward exploitation.

    Attributes:
        env: The Gymnasium environment to interact with.
        n_actions (int): Number of discrete actions in the environment.
        alpha (float): Per-tiling learning rate (the base learning rate divided by
            the number of tilings, so the combined update across all active tiles
            matches the intended step size).
        gamma (float): Discount factor for future rewards.
        epsilon (float): Current exploration probability for the ε-greedy policy.
        tc (TileCoder): Tile coder mapping continuous states to active tile indices.
        weights (defaultdict): Maps (action, *tile) keys to their learned weight,
            defaulting to 0.0 for tiles never seen before.
    """
    def __init__(self, env, tilings=8, bins=(8, 8), alpha=0.1, gamma=1.0, epsilon=1):
        self.env = env
        self.n_actions = env.action_space.n
        # Spread the learning rate across all tilings so that the combined update
        # over the (tilings) active features equals the intended step size.
        self.alpha = alpha / tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.tc = TileCoder(env.observation_space.low, env.observation_space.high, tilings, bins)
        # Sparse weight table; unseen (action, tile) keys default to 0.0.
        self.weights = defaultdict(float)

    def get_q(self, state, action):
        """
        Estimates the action-value Q(state, action) via the linear model.

        Since exactly one tile per tiling is active, Q is the sum of the weights of
        the active (action, tile) features.

        Args:
            state (array-like): The continuous state to evaluate.
            action (int): The action whose value is requested.

        Returns:
            float: The estimated action-value Q(state, action).
        """
        features = self.tc.get_features(state)
        return sum(self.weights[(action,) + f] for f in features)

    def choose_action(self, state):
        """
        Selects an action using the ε-greedy policy.

        With probability ε a random action is sampled (exploration); otherwise the
        action with the highest estimated Q-value is chosen (exploitation).

        Args:
            state (array-like): The current continuous state.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = [self.get_q(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action):
        """
        Applies one on-policy SARSA update to the weights of the active tiles.

        Computes the TD error using the value of the actually chosen next action
        (on-policy) and distributes the correction across the active features of
        the current state-action pair.

        Args:
            state (array-like): The state in which the action was taken.
            action (int): The action taken in ``state``.
            reward (float): The reward received after taking the action.
            next_state (array-like): The resulting next state.
            next_action (int): The action chosen in ``next_state`` (on-policy).
        """
        features = self.tc.get_features(state)
        q_current = self.get_q(state, action)
        q_next = self.get_q(next_state, next_action)
        # TD error: reward plus discounted value of the next (on-policy) action,
        # minus the current estimate.
        delta = reward + self.gamma * q_next - q_current
        # Each active tile contributes equally; nudge all of them toward the target.
        for f in features:
            self.weights[(action,) + f] += self.alpha * delta

    def train(self, num_epsiodes):
        """
        Trains the agent over a number of episodes using SARSA.

        Each episode runs until termination or truncation. Before every episode the
        exploration rate ε is decayed geometrically (down to a floor of 0.01), so
        the policy gradually shifts from exploration to exploitation.

        Args:
            num_epsiodes (int): The number of training episodes to run.

        Returns:
            list of float: The total (undiscounted) reward collected in each episode.
        """
        returns = []
        for episode in range(num_epsiodes):
            # Decay exploration toward a small floor before each episode.
            self.epsilon = max(0.01, self.epsilon * 0.995)
            state, _ = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0

            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # SARSA is on-policy: pick the next action, then update toward it.
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                total_reward += reward

            if episode % 50 == 49:
                print(f"Episode {episode + 1}: Total reward = {total_reward}")
            returns.append(total_reward)
        return returns

    def plot_learning_curve(self, rewards):
        """
        Plots the per-episode return together with a moving average.

        Args:
            rewards (list of float): The total reward per episode, e.g. the return
                value of :meth:`train`.
        """

        def moving_average(data, window_size=20):
            # Smooth the noisy per-episode returns to expose the learning trend.
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        plt.plot(rewards, label='Return per Episode', alpha=0.5)
        plt.plot(moving_average(rewards), label='Moving average', linewidth=2, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning  curve of SARSA with Tile Coding')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_episode(self, env, render=False, greedy=True):
        """
        Runs a single episode with the current policy (e.g. for evaluation or demo).

        Unlike :meth:`train`, this does not update any weights. The action can be
        chosen greedily (deterministic, for evaluation) or via the ε-greedy policy.

        Args:
            env: The environment to run the episode in (may differ from the training
                env, e.g. a render-enabled instance).
            render (bool): If True, render each step.
            greedy (bool): If True, always pick the highest-valued action;
                otherwise follow the ε-greedy policy.

        Returns:
            float: The total (undiscounted) reward collected during the episode.
        """
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            if greedy:
                # Select action greedily (deterministic)
                q_values = [self.get_q(state, a) for a in range(self.n_actions)]
                action = np.argmax(q_values)
            else:
                # Select action using ε-greedy strategy
                action = self.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        return total_reward