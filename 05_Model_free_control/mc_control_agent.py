import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class MCControlAgent:
    """
    Monte Carlo Control Agent for finding the optimal policy.

    Attributes:
        gamma (float): Discount factor for future rewards.
    """

    def __init__(self, gamma):
        """
        Initialize the Monte Carlo Control Agent.

        Args:
            gamma (float): Discount factor for future rewards.
        """
        self.gamma = gamma

    def gen_eps(self, env, policy):
        """
        Generate a single episode using the given policy.

        Args:
            env: The environment to generate the episode in.
            policy (function): A function mapping states to probabilities of actions.

        Returns:
            list: comprising a tuple of three lists (states, actions, rewards)
        """
        episode = []

        # YOUR CODE HERE

        return episode

    def on_policy_control(self, env, n_eps, policy, eps=1.0, eps_decay=0.9999):
        """
        Performs on-policy control using an ε-greedy strategy to improve the given policy.

        This method runs for a specified number of episodes, interacting with the environment
        while following and updating the provided ε-greedy policy. It estimates the action-value
        function Q and updates the policy accordingly. The ε value decays over time to reduce exploration.

        Args:
            env: The environment with which the agent interacts. Must follow the OpenAI Gym interface.
            n_eps (int): Number of episodes to run for policy improvement.
            policy (dict): A mapping from states to actions, which will be improved in place.
            eps (float, optional): Initial exploration rate (default is 1.0).
            eps_decay (float, optional): Multiplicative factor to decay ε after each episode (default is 0.9999).

        Returns:
            Q (defaultdict): A dictionary mapping state-action pairs to estimated action values.
            policy (dict): The improved policy after running the control algorithm.
        """

        Q = defaultdict(float)
        N = defaultdict(int)

        # YOUR CODE HERE

        return Q, policy

    def plot_policy(self, p):
        """
        Visualizes the learned policy for a Blackjack agent with and without a usable ace.

        Parameters:
        -----------
        p : np.ndarray
            A 3D NumPy array of shape (player_sum, dealer_card, usable_ace, action_probabilities),
            where p[:, :, 0] represents the policy when no usable ace is available, and
            p[:, :, 1] when a usable ace is available. The policy is assumed to contain
            action probabilities, and the best action is selected via argmax.

        Displays:
        ---------
        A matplotlib figure with two subplots:
            - Left: Optimal action policy without a usable ace
            - Right: Optimal action policy with a usable ace
        """
        p_no_ace = p[:, :, 0]
        p_have_ace = p[:, :, 1]

        best_policy_no_ace = np.argmax(p_no_ace, axis=2)
        best_policy_have_ace = np.argmax(p_have_ace, axis=2)

        fig, ax = plt.subplots(ncols=2, figsize=(9, 9))

        ax1, ax2 = ax

        m1 = ax1.matshow(best_policy_no_ace)
        m2 = ax2.matshow(best_policy_have_ace)

        xticks = np.arange(11, 22)
        yticks = np.arange(1, 11)
        # Show all ticks, remove what rows and columns to not to show
        ax1.set_yticks(xticks)
        ax1.set_xticks(yticks)
        ax2.set_yticks(xticks)
        ax2.set_xticks(yticks)

        ax1.set_xlim(.5, 10.5)
        ax2.set_xlim(.5, 10.5)
        ax1.set_ylim(11, 22)
        ax2.set_ylim(11, 22)

        ax1.set_ylabel('Player sum', fontsize=16)
        ax1.set_xlabel('Dealer showing card', fontsize=16)
        ax2.set_ylabel('Player sum', fontsize=16)
        ax2.set_xlabel('Dealer showing card', fontsize=16)

        ax1.set_title('Policy, no usable ace', fontsize=18)
        ax2.set_title('Policy, with usable ace', fontsize=18)

        fig.tight_layout(pad=4.0)

        fig.colorbar(m1, ax=ax, shrink=1, location='bottom')

        plt.show()

    def plot_blackjack(self, env, p, Q, ax1, ax2):
        """
        Plots the state-value function for the Blackjack environment using the current policy and action-value estimates.

        Parameters:
        -----------
        env : gym.Env
            The Blackjack environment instance, used to access the action space.

        p : np.ndarray
            A 3D array representing the policy. p[state][action] gives the probability
            of selecting an action in a given state, split by usable ace.

        Q : dict
            A dictionary mapping (state, action) pairs to estimated action-values.

        ax1 : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis object for plotting state-values with a usable ace.

        ax2 : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis object for plotting state-values without a usable ace.

        Displays:
        ---------
        Two 3D surface plots:
            - ax1: Value function with a usable ace
            - ax2: Value function without a usable ace
        """
        V = defaultdict(float)
        for i in range(11, 22):
            for j in range(1, 11):
                for k in [True, False]:
                    V[(i, j, k)] = sum([p[i, j, int(k)][a] * Q[((i, j, k), a)] for a in range(env.action_space.n)])

        player_sum = np.arange(12, 21 + 1)
        dealer_show = np.arange(1, 10 + 1)
        usable_ace = np.array([False, True])
        state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

        for i, player in enumerate(player_sum):
            for j, dealer in enumerate(dealer_show):
                for k, ace in enumerate(usable_ace):
                    state_values[i, j, k] = V[player, dealer, ace]

        X, Y = np.meshgrid(dealer_show, player_sum)

        ax1.plot_wireframe(X, Y, state_values[:, :, 1])
        ax2.plot_wireframe(X, Y, state_values[:, :, 0])

        for ax in ax1, ax2:
            ax.set_zlim(-1, 1)
            ax.set_ylabel('player sum')
            ax.set_xlabel('dealer showing')
            ax.set_zlabel('state-value')
            ax.set_box_aspect([1.0, 1.0, 0.35])

