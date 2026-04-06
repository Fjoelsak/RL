import numpy as np

class MarkovDecisionProcess:

    def __init__(self, states, actions, rewards, transProbs, terminalStates, gamma, **kwargs):
        """
        Class represents a markov decision process

        :param states: List
            list of states
        :param actions: Dictionary
            for each state the possible actions are provided as a list
        :param rewards: Dictionary
            rewards of each state-action pair
        :param transProbs: Dictionary
            probabilities of each state-action pair to the following state in the form
            {(0,1): {0: 0.1, 1: 0.8, 2: 0.1}, 1: {0: 0.3}}, that is, you arrive with 10% from
            state 0 taking action 1 in state 0
        :param terminalStates: List
            list of terminal states
        :param gamma: float
            discount factor
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminalStates = terminalStates
        self.transProbs = transProbs
        self.gamma = gamma

        self.state_values = kwargs.get('state_values',[0 for state in self.states])
        self.policy = kwargs.get('policy',
                                 {state: {action: 1 / len(self.actions[state]) for action in self.actions[state]} for
                                  state in states})

        print(self.policy)

        self.trans_prob_matrix = np.zeros((len(states), len(states)))

        self.states_without_terminal_state = [state for state in states if state not in terminalStates]

        for terminal_state in self.terminalStates:
            if self.state_values[terminal_state] != 0:
                raise Exception('Value of terminal state must be 0')

    def getPossibleNextStates(self, state: int, action: int) -> list:
        """
        Provides a list of possible successor states

        :param state: current state
        :return: list of names of possible next states
        """
        return list(self.transProbs[(state,action)].keys())

    def sample(self, startstate: int):
        """
        Samples states in a MDP until a terminal state is reached and returns
        the trajectory.

        :param startstate: state to start sampling from the episode
        :return: list of tuples (s_t, a_t, r_t+1) running through in an episode
        """
        traj = []
        curState = startstate
        while curState not in self.terminalStates:
            action = np.random.choice(self.actions[curState], p=list(self.policy[curState].values()))
            pos_successors = self.getPossibleNextStates(curState, action)
            probs = list(self.transProbs[(curState,action)][succ] for succ in pos_successors)
            traj.append((self.states[curState], int(action), self.rewards[(curState,action)]))
            curState = np.random.choice(pos_successors, p=probs)
        return traj

    def analytical_sol(self):
        """
        Calculates the state-value function analytically by reducing the MDP to an MRP for a given policy

        :return: V, a list with a value for each state
        """
        V = []

        # Calculate the vector R^pi by averaging over all actions in a certain state
        avg_rew = []
        for state in self.states_without_terminal_state:
            tmp = 0
            for stateaction in self.rewards.keys():
                if stateaction[0] == state:
                    tmp += self.policy[state][stateaction[1]]*self.rewards[stateaction]
            avg_rew.append(tmp)
        avg_rew.append(0)

        avg_prob = {}
        for state in self.states_without_terminal_state:
            for stateaction in self.transProbs.keys():
                if stateaction[0] == state:
                    for nextstate in self.transProbs[stateaction].keys():
                        if (state, nextstate) in avg_prob.keys():
                            avg_prob[(state, nextstate)] += self.policy[state][stateaction[1]] * self.transProbs[stateaction][nextstate]
                        else:
                            avg_prob[(state, nextstate)] = self.policy[state][stateaction[1]] * \
                                                            self.transProbs[stateaction][nextstate]

        avg_prob_matrix = np.zeros((len(self.states), len(self.states)))
        for ssnext in avg_prob.keys():
            avg_prob_matrix[ssnext[0], ssnext[1]] = avg_prob[ssnext]

        return np.dot(np.linalg.inv(np.eye(len(self.states)) - self.gamma * avg_prob_matrix), np.array(avg_rew))


