import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from collections import deque
from gymnasium import Env

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os


class ReinforceAgent:
    """
    Class for the REINFORCE agent using TensorFlow 2.15
    """

    def __init__(self, env, config):
        """
        Initializes the REINFORCE agent with a policy network and relevant training configuration.

        Parameters:
        ----------
        env : gymnasium.Env
            The environment the agent will interact with. Must have discrete action space and a flat observation space.

        config : dict
            Configuration dictionary containing:
                - 'EPISODES' (int): Number of training episodes.
                - 'MAX_TIMESTEPS' (int): Maximum timesteps per episode.
                - 'DISCOUNT' (float): Discount factor (gamma) for future rewards.
                - 'LEARNING_RATE' (float): Learning rate for the policy network optimizer.
                - 'HIDDEN_DIM' (int): Number of hidden units in the policy network.
                - 'LOGS' (bool): Flag to enable/disable logging of training statistics.

        Initializes:
        -----------
        - Policy network and optimizer.
        - Episode-specific storage for states, actions, and rewards.
        - Logging containers for performance metrics across episodes.
        """

        self.max_epsiodes = config['EPISODES']
        self.max_timesteps = config['MAX_TIMESTEPS']
        self.gamma = config['DISCOUNT']
        self.learning_rate = config['LEARNING_RATE']
        self.hidden_dim = config['HIDDEN_DIM']
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.logs = config['LOGS']

        self.policy_net = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Storage for one episode
        self.states = []
        self.actions = []
        self.rewards = []

        # information being stored after each episode
        self.reward_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []

    def _build_model(self):
        """
        Builds the policy network model for the REINFORCE agent.

        Returns:
        --------
        model : keras.Model
            A feedforward neural network with one hidden layer and softmax output,
            representing the agent's stochastic policy.

        Architecture:
        -------------
        - Input layer: Dimension matches the environment's observation space.
        - Hidden layer: Fully connected with ReLU activation (size = self.hidden_dim).
        - Output layer: Fully connected with softmax activation to represent a probability distribution over actions.
        """
        inputs = Input(shape=(self.input_dim,))
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        outputs = Dense(self.output_dim, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs)

    def get_action(self, state):
        """
        Samples an action from the current policy given the environment state.

        Parameters:
        ----------
        state : np.ndarray
            The current state of the environment (shape: [input_dim]).

        Returns:
        -------
        action : int
            The action sampled from the policy's probability distribution.

        action_prob : float
            The probability assigned by the policy to the sampled action.
        """
        state = np.expand_dims(state, axis=0)  # shape: (1, input_dim)
        action_probs = self.policy_net(state, training=False).numpy()[0]
        action = np.random.choice(self.output_dim, p=action_probs)
        return action, action_probs[action]

    def store_transition(self, state, action, reward):
        """
        Stores a single transition (state, action, reward) from the current episode.

        These stored transitions will be used later to compute returns and update the policy.

        Parameters:
        ----------
        state : np.ndarray
            The observed state at the current timestep.

        action : int
            The action taken in the current state.

        reward : float
            The reward received after taking the action.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def _compute_discounted_rewards(self):
        """
        Computes the discounted rewards for the current episode.

        Applies the discount factor γ to future rewards to compute the return for each timestep,
        and normalizes the resulting values for improved training stability.

        Returns:
        -------
        discounted : np.ndarray
            An array of normalized, discounted returns with the same length as the episode,
            where each element corresponds to the return from that timestep onward.
        """
        discounted = np.zeros_like(self.rewards, dtype=np.float32)
        cumulative = 0.0
        for t in reversed(range(len(self.rewards))):
            cumulative = self.rewards[t] + self.gamma * cumulative
            discounted[t] = cumulative
        # Normalize
        discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-9)
        return discounted

    def _update_policy(self, episode: int):
        """
        Updates the policy network using the REINFORCE algorithm.

        Computes the policy gradient based on the collected episode data
        and applies the gradient update to improve the policy.

        Steps:
        ------
        1. Compute normalized, discounted rewards.
        2. Calculate the log-probabilities of the taken actions.
        3. Compute the REINFORCE loss: -sum(log(pi(a|s)) * G_t).
        4. Apply gradients to update the policy network.
        5. Clear the stored episode transitions.

        Parameters:
        ----------
        episode : int
            The current training episode (used for logging or tracking progress).
        """
        discounted_rewards = self._compute_discounted_rewards()

        with tf.GradientTape() as tape:
            state_tensor = tf.convert_to_tensor(np.vstack(self.states), dtype=tf.float32)
            action_probs = self.policy_net(state_tensor, training=True)

            indices = np.arange(len(self.actions))
            chosen_action_probs = tf.gather_nd(action_probs,
                                               indices=np.vstack((indices, self.actions)).T)

            # +1e-9 avoids log(0) values causing NaNs
            log_probs = tf.math.log(chosen_action_probs + 1e-9)
            loss = -tf.reduce_sum(log_probs * discounted_rewards)

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        # Reset episode data
        self.states, self.actions, self.rewards = [], [], []

    def train(self, env: Env):
        """
        Trains the policy network using the REINFORCE algorithm over multiple episodes.

        For each episode:
        - Interacts with the environment to collect (state, action, reward) transitions.
        - Computes the policy gradient based on the episode's trajectory.
        - Updates the policy to maximize expected returns.
        - Tracks training metrics like total reward, timesteps, and moving average.

        Periodically saves model checkpoints and logs training progress.

        Parameters:
        ----------
        env : gymnasium.Env
            The environment in which the agent is trained. Must follow the Gymnasium API.
        """
        all_rewards = []
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.max_epsiodes):
            state, _ = env.reset()
            total_reward = 0
            timestep_per_episode = 0

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
                if (total_reward > max_reward):
                    self.save_model(self.logs+'/'+str(total_reward)+"_agent")
                elif (episode % 50 == 0):
                    self.save_model(self.logs+"/Episode_"+str(episode)+"_agent")
                    print('Episode:\t', episode, '\t Average Score:\t',np.mean(scores_deque))

            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}")

            # bookkeeping
            scores_deque.append(total_reward)
            max_reward = max(total_reward, max_reward)

            self.reward_episodes.append(total_reward)
            self.timesteps_per_episode.append(timestep_per_episode)
            self.average_score_100_episodes.append(np.mean(scores_deque))

        # Save all the information during training in a pandas dataframe and save it as a csv file
        self.save_data(self.logs, 'results.csv')

    def save_model(self, name):
        """
        Saves the weights of the policy network to a file.

        Parameters:
        ----------
        name : str
            The base filename (without extension) to use when saving the model.
            The final file will be saved as '<name>.keras'.
        """
        self.policy_net.save(name+".keras")

    def load_model(self, name):
        """
        Loads the weights of a previously saved policy network.

        Parameters:
        ----------
        name : str
            The base filename (without extension) from which to load the model.
            The method expects a file named '<name>.keras'.
        """
        self.policy_net.load_weights(name+".keras")

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):
        """
        Saves training statistics to a CSV file for later analysis.

        Parameters:
        ----------
        logdir : str
            Directory where the CSV file will be saved.

        name : str
            Filename of the CSV file.

        col_reward : str
            Column name for the list of total episode rewards.

        col_timesteps : str
            Column name for the list of episode lengths.

        col_average_score : str
            Column name for the moving average score over the last 100 episodes.
        """

        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})

        df.to_csv(logdir + '/' + name)

def plot_trainingsinformation(data,
                              data_names,
                              colors,
                              figsize=(15, 4),
                              ylim=3000,
                              columns=['Rewards', 'Timesteps per episode', 'Average score over 100 episodes'],
                              smoothing_factor=0.05,
                              alpha_non_smooth=0.3):
    """
    Plots key training information for one or more agents over time.

    Parameters:
    ----------
    data : list of pd.DataFrame
        A list containing pandas DataFrames, each representing training metrics for a different agent.
    data_names : list of str
        List of labels corresponding to each DataFrame in `data`.
    colors : list of str
        List of colors used for plotting each agent's data.
    figsize : tuple, optional (default=(15, 4))
        Size of the entire plot figure.
    ylim : int or float, optional (default=3000)
        Upper limit for the y-axis in the rewards plot.
    columns : list of str, optional
        Names of the columns to be plotted from the DataFrames.
        Should include ['Rewards', 'Average score over 100 episodes', 'Epsilon over episodes'].
    smoothing_factor : float, optional (default=0.05)
        Smoothing factor for the exponential weighted moving average.
        Applied to all columns except for 'Epsilon over episodes'.
    alpha_non_smooth : float, optional (default=0.3)
        Transparency level for the unsmoothed lines.

    Returns:
    -------
    None
        Displays the plot with training metrics.
    """
    # Create a subplot with one row and as many columns as specified
    fig, ax = plt.subplots(figsize=figsize, ncols=len(columns), nrows=1)

    for i, col in enumerate(columns):
        # Set full opacity for the Epsilon plot
        alpha = alpha_non_smooth if i != 2 else 1

        # Plot the original (unsmoothed) values with reduced opacity
        for k, df in enumerate(data):
            sns.lineplot(df, x=df.index, y=col, alpha=alpha, color=colors[k], ax=ax[i])

        # Apply smoothing and plot if not the Epsilon column
        if i != 2:
            for k, df in enumerate(data):
                sns.lineplot(df.ewm(alpha=smoothing_factor).mean(), x=df.index, y=col,
                             label=data_names[k], color=colors[k], ax=ax[i])

        # Adjust grid and x-axis label
        ax[i].grid(alpha=0.3)
        ax[i].set_xlabel('Episodes')

        # Adjust legend
        if i == 0:
            ax[i].get_legend().remove()
        else:
            ax[1].legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncols=len(columns))

    # Set y-axis limit for the Rewards plot
    ax[0].set_ylim(0, ylim)
