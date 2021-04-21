import numpy as np

from parameter_tracker import ParameterTracker


# This code will ensure that all actions a <= A will be executed in the first |A| steps, then reverts
# to e_greedy.
class Greedy:
    def __init__(self, random: np.random.RandomState, n_actions):
        self.random = random
        self.n_actions = n_actions
        self.initial_actions = self.random.permutation(range(n_actions))

    def next(self, q, epsilon):
        if self.initial_actions.size > 0:
            action = self.initial_actions[0]
            self.initial_actions = self.initial_actions[1:]
            return action
        return e_greedy(self.random, q, self.n_actions, epsilon)


# using egreedy policy to select actions
def e_greedy(seed: np.random.RandomState, q, actions, epsilon):
    if seed.uniform(0, 1) < (1 - epsilon):
        return seed.choice(np.flatnonzero(q == q.max()))
    else:
        return seed.choice(actions)


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))  # initializing q as a 0 array
    tracker = ParameterTracker()
    log = Logger.with_max_index(max_episodes, 0.05, level='info')

    for i in range(max_episodes):
        s = env.reset()
        a = e_greedy(random_state, q[s], env.n_actions, epsilon[i])  # action is selected based on egreedy
        terminal = False

        while not terminal:
            next_s, r, terminal = env.step(a)  # stores next state, reward and terminal
            next_a = e_greedy(random_state, q[s], env.n_actions, epsilon[i])  # using egreedy to find out next action
            q[s, a] = q[s, a] + eta[i] * (r + (gamma * q[next_s, next_a]) - q[s, a])  # storing values in q
            s = next_s  # stores next state as current state
            a = next_a  # stores next action as current action

        tracker.track(q)

    policy = q.argmax(axis=1)
    value = np.max(q, axis=1)

    # ## Uncomment this to look at the output from the parameter tracker ...
    # The most optimal policy has the highest weight. I can return to it
    # print(f'Tracking: {tracker}')

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))  # initializing q as a 0 array
    tracker = ParameterTracker()

    for i in range(max_episodes):
        s = env.reset()
        terminal = False

        while not terminal:
            a = e_greedy(random_state, q[s], env.n_actions, epsilon[i])  # using egreedy to find out next action
            next_s, r, terminal = env.step(a)  # stores next state, reward and terminal
            q[s, a] = q[s, a] + eta[i] * (r + (gamma * max(q[next_s])) - q[s, a])  # storing values in q
            s = next_s  # stores next state as current state
        tracker.track(q)

    policy = q.argmax(axis=1)
    value = np.max(q, axis=1)

    return policy, value
