import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from TileCoding import TileCoder

class LinearSarsaAgent:
    def __init__(self, env, tilings=8, bins=(8, 8), alpha=0.1, gamma=1.0, epsilon=1):
        self.env = env
        self.n_actions = env.action_space.n
        self.alpha = alpha / tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.tc = TileCoder(env.observation_space.low, env.observation_space.high, tilings, bins)
        self.weights = defaultdict(float)

    def get_q(self, state, action):
        features = self.tc.get_features(state)
        return sum(self.weights[(action,) + f] for f in features)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = [self.get_q(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action):
        features = self.tc.get_features(state)
        q_current = self.get_q(state, action)
        q_next = self.get_q(next_state, next_action)
        delta = reward + self.gamma * q_next - q_current
        for f in features:
            self.weights[(action,) + f] += self.alpha * delta

    def train(self, num_epsiodes):
        returns = []
        for episode in range(num_epsiodes):
            self.epsilon = self.epsilon / (episode+1)
            state, _ = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0

            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                total_reward += reward

            if episode % 50 == 49:
                print(f"Episode {episode + 1}: Total reward = {total_reward}")
            returns.append(total_reward)
        return returns

    def plot_learning_curve(self, rewards):

        def moving_average(data, window_size=20):
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
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            if greedy:
                # Wähle Aktion deterministisch (greedy)
                q_values = [self.get_q(state, a) for a in range(self.n_actions)]
                action = np.argmax(q_values)
            else:
                # Wähle Aktion mit ε-Greedy-Strategie
                action = self.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        return total_reward