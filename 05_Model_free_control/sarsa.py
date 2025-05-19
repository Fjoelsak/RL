import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def run_episode(env, policy=None, render=True):
    """ Follow policy through an environment's episode and return an array of collected rewards """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    state = env.reset()
    if render:
        env.render()

    done = False
    rewards = []
    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.argmax(policy[state_ridx])
        state, reward, done, truncated, info = env.step(action)
        rewards += [reward]

        if render:
            env.render()

    if render:
        import matplotlib.pyplot as plt
        plt.show()

    return rewards

def sarsa(env, num_episodes, eps0=0.5, alpha=0.5):
    """ On-policy Sarsa algorithm (with exploration rate decay) """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # Number of available actions and maximal state ravel index
    n_action = env.action_space.n
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1

    # Initialization of action value function
    q = np.zeros([n_state_ridx, n_action], dtype=float)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state_ridx, n_action], dtype=float) / n_action
    
    history = [0] * num_episodes
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        # take action derived from state s with eps-greedy policy
        action = np.random.choice(n_action, p=policy[state_ridx])

        done = False
        while not done:
            # Step the environment forward and check for termination
            next_state, reward, done, truncated, info = env.step(action)
            next_state_ridx = np.ravel_multi_index(next_state, env.observation_space.nvec)
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])

            # Update q values
            """ TODO """
            #q[state_ridx, action] += ??
            
            # Extract eps-greedy policy from the updated q values
            """ TODO: epsilon und policy anpassen bezüglich der neuen q Werte """
            #eps = ??
            #policy[??, ??] = ??
            
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update
            state_ridx = next_state_ridx
            action = next_action
            history[episode] += 1

    return q, policy, history

def plot_results(env, q, policy):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Optimal Value Function and Policy")

    q = np.copy(q)
    unvisited = np.where(q == 0)
    q[unvisited] = -np.inf
    v = np.max(q, axis=1).reshape(env.observation_space.nvec)
    ax.imshow(v.T, origin='lower')

    a_stars = np.argmax(policy, axis=1)
    arrows = np.array([env.unwrapped.actions[a] for a in a_stars])
    arrows[unvisited[0], :] = 0
    arrows = arrows.reshape([*env.observation_space.nvec, 2])
    xr = np.arange(env.observation_space.nvec[0])
    yr = np.arange(env.observation_space.nvec[1])
    ax.quiver(xr, yr, arrows[:, :, 0].T, arrows[:, :, 1].T, pivot='mid')