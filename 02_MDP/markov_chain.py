import numpy as np

class MarkovChain:

    def __init__(self, states: list, terminalStates: list, transProbs: dict):
        """
        Class represents a markov chain

        :param states: List
            list of states
        :param terminalStates: List
            list of terminal states
        :param probs: Dictionary
            probabilities of each state to the following state in the form
            {0: {0: 0.1, 1: 0.8, 2: 0.1}, 1: {0: 0.3}}, that is, you arrive with 10% from
            state 0 to state 1
        """
        self.states = states
        self.terminalStates = terminalStates
        self.transProbs = transProbs

        self.trans_prob_matrix = np.zeros((len(states), len(states)))

        for cs in transProbs.keys():
            for sn in transProbs[cs].keys():
                self.trans_prob_matrix[cs, sn] = transProbs[cs][sn]

        for ts in terminalStates:
            self.trans_prob_matrix[ts, ts] = 1.0

    def getNextState(self, state: int) -> tuple:
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
        :return: list of names of states running through in an episode
        """
        traj = []
        curState = startstate
        traj.append(self.states[curState])
        while curState not in self.terminalStates:
            pos_successors = self.getNextState(curState)
            probs = list(self.transProbs[curState][succ] for succ in pos_successors)
            curState = np.random.choice(pos_successors, p=probs)
            traj.append(self.states[curState])
        return traj

    def paging(self, startvec: np.ndarray, maxTimesteps: int):
        """
        Calculates the probabilites of being in a certain state after maxTimesteps

        :param startvec: vector of starting state
        :return:    vector with probabilities
        """
        pass