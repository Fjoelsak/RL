import numpy as np

class MarkovRewardProcess:

    def __init__(self, states, rewards, transProbs, terminalStates, gamma, **kwargs):
        """
        Class represents a markov reward process

        :param states: List
            list of states
        :param rewards: List
            list of rewards for each state
        :param transProbs: Dictionary
            probabilities of each state to the following state in the form
            {0: {0: 0.1, 1: 0.8, 2: 0.1}, 1: {0: 0.3}}, that is, you arrive with 10% from
            state 0 to state 1
        :param terminalStates: List
            list of terminal states
        :param gamma: float
            discount factor
        """
        self.states = states
        self.rewards = rewards
        self.terminalStates = terminalStates
        self.transProbs = transProbs
        self.gamma = gamma

        self.trans_prob_matrix = np.zeros((len(states), len(states)))

        for cs in transProbs.keys():
            for sn in transProbs[cs].keys():
                self.trans_prob_matrix[cs, sn] = transProbs[cs][sn]

        for ts in terminalStates:
            self.trans_prob_matrix[ts, ts] = 1.0


    def getPossibleNextStates(self, state: int) -> list:
        """
        Provides a list of possible successor states

        :param state: current state
        :return: list of names of possible next states
        """
        return list(self.transProbs[state].keys())

    def sample(self, startstate: int):
        """
        Samples states in a markov chain until a terminal state is reached and returns
        the trajectory.

        :param startstate: state to start sampling from the episode
        :return: list of names of states running through in an episode and calculated return
        """
        traj, rew = [], []
        curState = startstate
        traj.append(self.states[curState])
        rew.append(self.rewards[curState])
        while curState not in self.terminalStates:
            pos_successors = self.getPossibleNextStates(curState)
            probs = list(self.transProbs[curState][succ] for succ in pos_successors)
            curState = np.random.choice(pos_successors, p=probs)
            traj.append(self.states[curState])
            rew.append(self.rewards[curState])
        return traj, self.calc_return(rew)

    def calc_return(self, rewards):
        """
        Calculates the return of the given rewards after an episode

        :param rewards: list of rewards gathered during an episode
        :return: calculated return
        """
        pass

    def analytical_sol(self):
        """
        calculates the analytical solution of the value function according to the lecture slides

        :return: a list of values for each state
        """
        pass

