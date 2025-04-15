import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class MCEvalAgent:
    """
    Monte Carlo Evaluation Agent for estimating state values under a given policy.

    Attributes:
        gamma (float): Discount factor for future rewards.
    """

    def __init__(self, gamma):
        """
        Initialize the Monte Carlo Evaluation Agent.

        Args:
            gamma (float): Discount factor for future rewards.
        """
        self.gamma = gamma

    def gen_eps(self, env, policy):
        """
        Generate a single episode using the given policy.

        Args:
            env: The environment to generate the episode in.
            policy (function): A function mapping states to actions.

        Returns:
            tuple: A tuple of three lists (states, actions, rewards)
        """
        states, actions, rewards = [], [], []

        # YOUR CODE HERE

        return states, actions, rewards

    def eval(self, env, n_episodes, policy):
        """
        Evaluate the given policy over multiple episodes using first-visit Monte Carlo method.

        Args:
            env: The environment to evaluate in. Must follow OpenAI Gym API.
            n_episodes (int): Number of episodes to simulate.
            policy (function): A function mapping states to actions.

        Returns:
            defaultdict: A mapping from states to estimated state-values.
        """
        # Initialize value table and visit count dictionaries
        # States are used as keys, values as floats or ints
        # defaultdicts are used for clean dynamic table use without key errors
        value_table = defaultdict(float)
        visit_count = defaultdict(int)

        """ 
        Simulate episodes, calculate mean returns and update the value function
        """
        # YOUR CODE HERE

        return value_table


    def plot_blackjack(self, V):
        """
        Plot the state-value function for Blackjack.

        Args:
            V (dict): A dictionary mapping (player_sum, dealer_showing, usable_ace) to state values.
        """
        fig, axes = plt.subplots(nrows=2, figsize=(4, 10), subplot_kw={'projection': '3d'})
        axes[0].set_title('Value function without usable ace')
        axes[1].set_title('Value function with usable ace')

        player_sum = np.arange(12, 22)
        dealer_show = np.arange(1, 11)
        usable_ace = np.array([False, True])

        state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

        for i, player in enumerate(player_sum):
            for j, dealer in enumerate(dealer_show):
                for k, ace in enumerate(usable_ace):
                    state_values[i, j, k] = V[player, dealer, ace]

        X, Y = np.meshgrid(dealer_show, player_sum)

        axes[0].plot_wireframe(X, Y, state_values[:, :, 1])  # with usable ace
        axes[1].plot_wireframe(X, Y, state_values[:, :, 0])  # without usable ace

        for ax in axes:
            ax.set_zlim(-1, 1)
            ax.set_ylabel('Player sum')
            ax.set_xlabel('Dealer showing')
            ax.set_zlabel('State-value')

        plt.show()
