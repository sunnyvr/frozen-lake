import numpy as np

from environment import Environment
from logger import Logger
from tabular_modelfree_reinforcement import e_greedy


class LinearWrapper:
    def __init__(self, env: Environment):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        self.state_action_shape = (self.n_states, self.n_actions)

    @property
    def state(self):
        return self.env.state

    def identify_state(self, next_features):
        return next_features[0].reshape(self.state_action_shape)[:, 0].argmax()

    def feature_index(self, state, action):
        return np.ravel_multi_index((state, action), (self.n_states, self.n_actions))

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = self.feature_index(s, a)
            features[a, i] = 1.0
        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    n_actions = env.n_actions

    theta = np.zeros(env.n_features)

    # Set debugging level to 'trace', 'info', 'debug' or 'warn' or 'off
    log = Logger.with_max_index(max_episodes, 0.05, level='off')

    for episode in range(max_episodes):
        log.set_index(episode)

        features = env.reset()
        Q = features @ theta

        log.info(f'Episode {episode}, Q = {Q}')

        # 1: Select action
        action = e_greedy(random, Q, n_actions, epsilon[episode])

        terminal = False
        while not terminal:
            # 2: Observed reward and next state for action
            next_features, reward, terminal = env.step(action)
            # 3: Select action a' for s' greedy on Q
            next_action = e_greedy(random, Q, n_actions, epsilon[episode])

            # 4: Update weights
            theta, Q = update_weights(lambda q: q[next_action], theta, Q, action, reward, features, next_features,
                                      gamma,
                                      eta[episode])

            # 5: Update action and state
            action = next_action
            features = next_features

    return theta


def update_weights(q_maximizer, theta, Q, action, reward, features, next_features, gamma, eta):
    delta = reward - Q[action]
    Q = next_features @ theta
    delta += gamma * q_maximizer(Q)
    theta = theta + eta * delta * features[action]
    return theta, Q


def theta_delta(action_maximizing_q, theta, Q, action, next_action, reward, features, next_features, gamma,
                episode_eta):
    delta = reward - Q[action]
    Q = next_features @ theta
    delta += gamma * action_maximizing_q(Q, next_action)
    return episode_eta * delta * features[action]


def linear_q_learning(env: LinearWrapper, max_episodes, eta, gamma, epsilon, seed=None):
    random = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # Step 0: Initialize theta
    theta = np.zeros(env.n_features)

    # Set debugging level to 'trace', 'info', 'debug' or 'warn' or 'off
    log = Logger.with_max_index(max_episodes, 0.05, level='off')

    for episode in range(max_episodes):
        log.set_index(episode)

        features = env.reset()
        # Step 1: Initial state of episode
        Q = features @ theta

        log.info(f'Episode {episode}: Q = {Q},  states, max Q approx = {np.max(theta)}')

        terminal = False
        while not terminal:
            # Step 2: choose action
            action = e_greedy(random, Q, env.n_actions, epsilon[episode])

            # Step 3: Take action, observe reward and next_state
            next_features, reward, terminal = env.step(action)

            # Step 4: Update weights
            theta, Q = update_weights(lambda q: q.max(), theta, Q, action, reward, features, next_features, gamma,
                                      eta[episode])

            # Step 5: Update state
            features = next_features

    return theta
