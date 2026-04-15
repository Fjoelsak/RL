import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import tensorflow_probability as tfp

from plot_utils import plot_trainingsinformation


class A2CAgent:
    def __init__(self, env, config):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]  # assume symmetric bounds

        self.gamma = config['DISCOUNT']
        self.lr_actor = config['LEARNING_RATE']
        self.lr_critic = config['LEARNING_RATE']
        self.hidden_dim = config['HIDDEN_DIM']
        self.batch_size = config['BATCH_SIZE']

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = Adam(learning_rate=self.lr_critic)

        self.max_episodes = config['EPISODES']
        self.max_timesteps = config['MAX_TIMESTEPS']
        self.logs = config['LOGS']

        self.reward_episodes = []
        self.timesteps_per_episode = []
        self.average_score_100_episodes = []

    def _build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        mu = Dense(self.action_dim, activation='tanh')(x)
        std = Dense(self.action_dim, activation='softplus')(x)
        return Model(inputs, [mu, std])

    def _build_critic(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        value = Dense(1, activation='linear')(x)
        return Model(inputs, value)

    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        mu, std = self.actor(state)
        mu, std = mu.numpy()[0], std.numpy()[0]
        action = np.random.normal(mu, std)
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def train(self):
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

                if t % self.batch_size == 0:
                    self.update_batch(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []

                state = next_state
                total_reward += reward
                if done:
                    break

            self.reward_episodes.append(total_reward)
            self.timesteps_per_episode.append(timestep)
            scores_deque.append(total_reward)
            self.average_score_100_episodes.append(np.mean(scores_deque))

            if (total_reward == max(scores_deque)):
                self.save_model(self.logs + f"/{total_reward:.2f}_agent")
            elif episode % 50 == 0:
                self.save_model(self.logs + f"/Episode_{episode}_agent")

            if episode % 50 == 0:
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Average(100): {np.mean(scores_deque):.2f}")

        self.save_data(self.logs, 'results.csv')

    def update_batch(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape(persistent=True) as tape:
            values = tf.squeeze(self.critic(states))
            next_values = tf.squeeze(self.critic(next_states))
            targets = rewards + (1 - dones) * self.gamma * next_values
            td_errors = targets - values

            mu, std = self.actor(states)
            dist = tfp.distributions.Normal(mu, std)
            log_probs = tf.reduce_sum(dist.log_prob(actions), axis=1)
            actor_loss = -tf.reduce_mean(log_probs * td_errors)
            critic_loss = tf.reduce_mean(tf.square(td_errors))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def save_model(self, name):
        self.actor.save(name + "_actor.keras")
        self.critic.save(name + "_critic.keras")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.keras")
        self.critic.load_weights(name + "_critic.keras")

    def save_data(self, logdir, name,
                  col_reward='Rewards',
                  col_timesteps='Timesteps per episode',
                  col_average_score='Average score over 100 episodes'):
        df = pd.DataFrame({col_reward: self.reward_episodes,
                           col_timesteps: self.timesteps_per_episode,
                           col_average_score: self.average_score_100_episodes})
        df.to_csv(logdir + '/' + name)

