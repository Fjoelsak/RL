import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class WindyGridworldEnv(gym.Env):
    """ Windy Gridworld environment """
    size = (10, 7)
    start = (0, 3)
    goal = (7, 3)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    
    # standard actions
    pawns_actions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    
    """ TODO: definition of advanced actions """
    # kings actions
    # kings_action = 
    
    # stop action
    # stop_action = 

    observation_space = gym.spaces.MultiDiscrete(size)
    reward_range = (-1, -1)

    def __init__(self, king=False, stop=False):
        self.king = king
        self.stop = stop

        # definition of possible actions
        self.actions = self.pawns_actions[:]
        
        """ TODO: Locations to place your code for adding advanced actions """
            
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.position = None
        self.arrow = None

        self.ax = None

    def step(self, action):
        assert self.action_space.contains(action)

        # Calculate move vector and new position
        delta = self.actions[action]
        position = self.position + np.array(delta)

        # Add wind to the position
        wind = self.wind[self.position[0]]
        position[1] += wind

        # Store position for the next step and calculate arrow for rendering
        position = np.clip(position, 0, self.observation_space.nvec - 1)
        self.arrow = position - self.position
        self.position = position

        # Check for terminal state
        done = (position == self.goal).all()
        reward = -1

        assert self.observation_space.contains(position)    
        return position, reward, done, False, {}

    def reset(self, seed = None, options = None):
        self.position = np.array(self.start)
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.position

    def render(self, mode='human'):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Background colored by wind strength
            wind = np.vstack([self.wind] * self.size[1])
            self.ax.imshow(wind, aspect='equal', origin='lower', cmap='Blues')

            # Annotations at start and goal positions
            self.ax.annotate("G", self.goal, size=25, color='gray', ha='center', va='center')
            self.ax.annotate("S", self.start, size=25, color='gray', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.wind)))
            self.ax.set_xticklabels(self.wind)
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.size[0]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.size[1]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)

        # Arrow pointing from the previous to the current position
        if (self.arrow == 0).all():
            patch = mpatches.Circle(self.position, radius=0.05, color='black', zorder=1)
        else:
            patch = mpatches.FancyArrow(*(self.position - self.arrow), *self.arrow, color='black',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
        self.ax.add_patch(patch)


gym.envs.registration.register(
    id='WindyGridworld-v0',
    entry_point=lambda king, stop: WindyGridworldEnv(king, stop),
    kwargs={'king': False, 'stop': False},
    max_episode_steps=5_000,
)