from abc import abstractmethod, ABC

import numpy as np


class EnvironmentModel(ABC):
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random = np.random.RandomState(seed)

    @property
    def terminal_state(self): return None

    @abstractmethod
    def p(self, next_state, state, action):
        raise NotImplementedError()

    @abstractmethod
    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        # print(f'n_states = {self.n_states}, p={p}')
        if sum(p) == 0:
            return state, 0
        next_state = self.random.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward
