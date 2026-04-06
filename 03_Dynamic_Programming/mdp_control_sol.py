import numpy as np
import time

class mdpControl:
    """
        A class to handle MDP (Markov Decision Process) control tasks such as policy evaluation
        and policy iteration using dynamic programming techniques.

        Attributes:
        ----------
        env : gym.Env
            The environment object for the MDP problem.
        nS : int
            The number of states in the environment.
        nA : int
            The number of actions in the environment.
    """

    def __init__(self, env, gamma = 0.99, theta = 1e-6):
        """
        Initializes the MDP control class with the environment.

        Parameters:
        ----------
        env : gym.Env
            The environment to interact with. It must have an observation space and action space.
        gamma : float, optional
            Discount factor, by default 0.99.
        """
        self.env = env
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.gamma = gamma
        self.theta = theta

    def policy_evaluation(self, policy):
        """
        Evaluates a given policy using the Bellman equation until convergence.

        Parameters:
        ----------
        policy : np.array of shape [nS, nA]
            The policy to evaluate, where each state has a probability distribution over actions.
        theta : float, optional
            Convergence threshold, by default 1e-6.

        Returns:
        -------
        V : np.array of shape [nS]
            The value function for the evaluated policy.
        """
        V = np.zeros(self.nS)
        while True:
            delta = 0
            for s in range(self.nS):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.unwrapped.P[s][a]:
                        v += action_prob * (reward + self.gamma * prob * V[next_state] * (not done))
                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < self.theta:
                break
        return V

    def policy_iteration(self):
        """
        Performs policy iteration by evaluating and improving the policy until it converges.

        Parameters:
        ----------
        gamma : float, optional
            Discount factor, by default 0.99.

        Returns:
        -------
        policy : np.array of shape [nS, nA]
            The optimal policy found after iteration.
        V : np.array of shape [nS]
            The value function of the optimal policy.
        """
        policy = np.ones([self.nS, self.nA]) / self.nA  # random policy
        i = 0
        while True:
            V = self.policy_evaluation(policy)
            policy_stable = True
            for s in range(self.nS):
                old_action = np.argmax(policy[s])
                action_values = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, next_state, reward, done in self.env.unwrapped.P[s][a]:
                        action_values[a] += (reward + self.gamma * prob * V[next_state] * (not done))
                best_action = np.argmax(action_values)
                if old_action != best_action:
                    policy_stable = False
                policy[s] = np.eye(self.nA)[best_action]
            i = i + 1
            if policy_stable:
                print(i)
                break
        return policy, V

    def value_iteration(self):
        """
        Performs value iteration to find the optimal value function and policy.

        Returns:
        -------
        policy : np.array of shape [nS, nA]
            The optimal policy derived from the value function.
        V : np.array of shape [nS]
            The optimal value function.
        """
        V = np.zeros(self.nS)
        i = 0
        while True:
            delta = 0
            for s in range(self.nS):
                action_values = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, next_state, reward, done in self.env.unwrapped.P[s][a]:
                        action_values[a] +=  (reward + self.gamma * prob * V[next_state] * (not done))
                max_value = np.max(action_values)
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
            i = i + 1
            if delta < self.theta:
                print(i)
                break

        policy = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            action_values = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, next_state, reward, done in self.env.unwrapped.P[s][a]:
                    action_values[a] +=  (reward + self.gamma * prob * V[next_state] * (not done))
            best_action = np.argmax(action_values)
            policy[s] = np.eye(self.nA)[best_action]

        return policy, V

    def render_single(self, policy, max_steps=100):
        """
          This function does not need to be modified
          Renders policy once on environment. Watch your agent play!

          Parameters
          ----------
          policy: np.array of shape [env.nS]
            The action to take at a given state
          max_steps: int, optional
            The maximum number of steps to take
        """
        episode_reward = 0
        ob, _ = self.env.reset()
        terminated, truncated = False, False

        for t in range(max_steps):
            self.env.render()
            time.sleep(0.25)
            a = np.argmax(policy[ob])
            ob, rew, terminated, truncated, _ = self.env.step(a)
            episode_reward += rew
            if terminated or truncated:
                break
        self.env.render()

        if not terminated or truncated:
            print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
        else:
            print("Episode reward: %f" % episode_reward)