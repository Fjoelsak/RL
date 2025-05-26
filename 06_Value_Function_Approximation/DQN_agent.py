
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    """
    In this class the DQN agent is implemented
    """
    def __init__(self, env, config):
                
        # used parameter within the agent
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
        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen = config['REPLAY_MEMORY_SIZE']) 
    
        # dimensions of the action and state space
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape
        
        # both q networks for online action choice and the target network
        self.model = self.create_model()
        self.targetmodel = self.create_model()
        self.targetmodel.set_weights(self.model.get_weights())
        
        # counter for training steps used for updating the target network from time to time (defined in config)
        self.counterDQNTrained = 0

        # information being stored after each epise
        self.reward_episodes = []
        self.epsilon_over_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []
    
    def create_model(self):
        ''' DQN definition, from 2 inputs to 2 hidden layers with 24, 48 nodes with relu activation function. 
        Output layer has 3 nodes with a linear activation function '''
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48, activation='relu')(state_h1)
        output = Dense(self.action_dim, activation='linear')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        # loss function as Mean Squared Error with an adam optimizer with given learning rate
        model.compile(optimizer=Adam(learning_rate=self.learningRate), loss='mse')
        return model

    def load_model(self, name):
        ''' loads a model, that is, the weights of the DQN function approximator '''
        self.model.load_weights(name+".h5")
        
    def save_model(self, name):
        ''' saves the weights of the DQN '''
        self.model.save(name+".h5")
    
    def memorize(self, state, action, reward, next_state, done):
        # Store the transition into the replay-memory
        self.replay_memory.append((state, action, reward, next_state, done))

    def trainDQN(self):
        self.counterDQNTrained += 1
        
        # sample minibatch
        batch = random.sample(self.replay_memory,self.miniBatchSize)

        X_cur_states = []
        X_next_states = []
        
        for index, sample in enumerate(batch):
            cur_state, action, reward, next_state, done = sample
            X_cur_states.append(cur_state)
            X_next_states.append(next_state)
        
        X_cur_states = np.array(X_cur_states)
        X_next_states = np.array(X_next_states)
        
        cur_action_values = self.model.predict(X_cur_states, verbose=0)
        next_action_values = self.targetmodel.predict(X_next_states, verbose=0)

        for index, sample in enumerate(batch):
            cur_state, action, reward, next_state, done = sample
            cur_action_values[index][action] = reward + (1-done) * self.discount * np.amax(next_action_values[index])

        # Gradient update each sample
        self.model.train_on_batch(X_cur_states, cur_action_values)
        
        # for each updateTQNW steps we adjust the DQN update the target network with the weights
        if self.counterDQNTrained % self.updateTQNW == 0:
            self.targetmodel.set_weights(self.model.get_weights())

    def train(self, env):
        ''' the actual training of the agent '''
        logdir = self.logs 

        # for data gathering
        max_reward = -999999
        scores_deque = deque(maxlen=100)

        for episode in range(self.episodes):
            cur_state,_ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            #Beschränkung des Trainings auf eine maximale Episodenlänge zur Beschleunigung des Trainings
            while not done:
                episode_length += 1
                # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower. 
                # Default is showing the agent after each 50 episodes
                if self.visualization:
                    env.render()

                if(np.random.uniform(0, 1) < self.epsilon):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward - Dies bleibt bei DDQN gleich!
                    action = np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0), verbose=0)[0])

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward

                if done:
                    if (episode_reward > max_reward):
                        self.save_model(logdir+'/'+str(episode_reward)+"_agent")
                    elif (episode % 50 == 0):
                        self.save_model(logdir+"/Episode_"+str(episode)+"_agent")
                        print('Episode:\t', episode, '\t Average Score:\t',np.mean(scores_deque))
                
                # Add experience to replay memory buffer
                self.memorize(cur_state, action, reward, next_state, done)
                cur_state = next_state
                
                # only train DQN if there are enough transitions stored
                if(len(self.replay_memory) < self.minReplayMem):
                    continue

                if episode_length % self.trainFrequency == 0:
                    self.trainDQN()

            print('episode: {}, reward: {}'.format(episode+1,episode_reward))

            # Decrease epsilon
            if(self.epsilon > self.minEps and len(self.replay_memory) > self.minReplayMem):
                self.epsilon *= self.epsDecay

            # some bookkeeping.
            scores_deque.append(episode_reward)
            max_reward = max(episode_reward, max_reward)
            
            # saving the important scalars
            self.reward_episodes.append(episode_reward)
            self.epsilon_over_episodes.append(self.epsilon)
            self.timesteps_per_episode.append(episode_length)
            self.average_score_100_episodes.append(np.mean(scores_deque))
        
        # Save all the information during training in a pandas dataframe and save it as a csv file
        self.save_data(logdir, 'results.csv')

    def save_data(self, logdir, name, 
                  col_reward = 'Rewards', 
                  col_epsilon = 'Epsilon over episodes', 
                  col_timesteps = 'Timesteps per episode',
                  col_average_score = 'Average score over 100 episodes'):
        
        df = pd.DataFrame({col_reward: self.reward_episodes, 
                           col_epsilon: self.epsilon_over_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})

        df.to_csv(logdir + '/' + name)

    def test(self, env, name, TOTAL_EPISODES  = 10):
        ''' load the weights of the DQN and perform 10 steps in the environment '''
        # create and load weights of the model
        self.load_model(name)

        # Number of episodes in which agent manages to won the game before time is over
        episodes_won = 0

        for _ in range(TOTAL_EPISODES):
            episode_reward = 0
            cur_state,_ = env.reset()
            done = False
            episode_len = 0
            while not done:
                env.render()
                episode_len += 1
                next_state, reward, terminated, truncated ,_ = env.step(np.argmax(self.model.predict(np.expand_dims(cur_state, axis=0), verbose=0)))
                done = terminated or truncated
                if done and episode_len > 475:
                    episodes_won += 1
                cur_state = next_state
                episode_reward += reward
            print('EPISODE_REWARD', episode_reward)
            
        print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')

def plot_trainingsinformation(data,
                              data_names,
                              colors,
                              figsize=(15, 4),
                              ylim=3000,
                              columns=['Rewards', 'Average score over 100 episodes', 'Epsilon over episodes'],
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
