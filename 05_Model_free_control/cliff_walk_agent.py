from math import gamma

import numpy as np

class CliffWalkAgent:

    def __init__(self, gamma):
        self.gamma = gamma

    def run_episode(self, env, Q, render=True, epsilon=0.0):
        """
        Runs a single episode using the given Q-table and epsilon-greedy policy.

        At each step, an action is selected using the epsilon-greedy strategy.
        The environment is updated accordingly until the episode terminates or is truncated.
        Optionally renders the environment at each step.

        Parameters
        ----------
        env : gym.Env
            The environment in which the episode is run.
        Q : np.ndarray
            The state-action value table used to guide action selection.
        render : bool, optional
            Whether to render the environment at each step (default is True).
        epsilon : float, optional
            The probability of choosing a random action (default is 0.0, i.e., greedy policy).

        Returns
        -------
        rewards : list of float
            A list of rewards obtained at each time step during the episode.
        """
        rewards = []

        state, prob = env.reset()

        if render:
            env.render()

        while True:
            action = self.epsilon_greedy(env, Q, state, epsilon)

            state, reward, terminated, truncated, _ = env.step(action)

            if render:
                env.render()

            rewards.append(reward)

            if terminated or truncated:
                break

        return rewards

    def epsilon_greedy(self, env, Q, state, epsilon):
        """
        Selects an action using the epsilon-greedy strategy.

        With probability `epsilon`, a random action is selected (exploration).
        Otherwise, the action with the highest estimated Q-value is chosen (exploitation).

        Parameters
        ----------
        env : gym.Env
            The environment, used to sample random actions.
        Q : np.ndarray
            The state-action value table.
        state : int or tuple
            The current state for which the action is to be selected.
        epsilon : float
            The probability of choosing a random action (0 <= epsilon <= 1).

        Returns
        -------
        action : int
            The selected action.
        """
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state, :])

    def sarsa(self, env, alpha=0.1, epsilon=0.1, eps_decay=0.99, n_episodes=1000):
        """
        Performs the SARSA algorithm (on-policy TD control) to learn a state-action value function.

        The agent interacts with the environment using an epsilon-greedy policy derived from the current Q-values.
        Updates are made using the SARSA update rule. Epsilon decays over time to gradually shift from exploration to exploitation.

        Parameters
        ----------
        env : gym.Env
            The environment in which the agent learns.
        alpha : float, optional
            The learning rate or step size (default is 0.1).
        epsilon : float, optional
            The initial exploration rate for the epsilon-greedy policy (default is 0.1).
        eps_decay : float, optional
            Multiplicative decay factor for epsilon after each step (default is 0.99).
        n_episodes : int, optional
            The number of episodes to train over (default is 1000).

        Returns
        -------
        Q : np.ndarray
            The learned state-action value table.
        rewards_per_episode : list of float
            Total reward obtained in each episode.
        """

        rewards_per_episode = []
        Q = np.zeros((env.observation_space.n, env.action_space.n))

        # TODO: your code

        return Q, rewards_per_episode

    def qlearning(self, env, alpha=0.1, epsilon=0.1, eps_decay=0.99, n_episodes=1000):
        """
        Performs the Q-Learning algorithm (off-policy TD control) to learn a state-action value function.

        The agent updates Q-values using the maximum future reward estimate (greedy action) rather than following the policy used for action selection.
        Exploration is controlled by an epsilon-greedy strategy, with epsilon decaying over time.

        Parameters
        ----------
        env : gym.Env
            The environment in which the agent learns.
        alpha : float, optional
            The learning rate or step size (default is 0.1).
        epsilon : float, optional
            The initial exploration rate for the epsilon-greedy policy (default is 0.1).
        eps_decay : float, optional
            Multiplicative decay factor for epsilon after each step (default is 0.99).
        n_episodes : int, optional
            The number of episodes to train over (default is 1000).

        Returns
        -------
        Q : np.ndarray
            The learned state-action value table.
        rewards_per_episode : list of float
            Total reward obtained in each episode.
        """

        rewards_per_epsisode = []
        Q = np.zeros((env.observation_space.n, env.action_space.n))

        # TODO: your code here

        return Q, rewards_per_epsisode
