from abc import ABC, abstractmethod
from typing import Callable, Any

import numpy as np

from environment_model import EnvironmentModel


class Environment(EnvironmentModel, ABC):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        super(Environment, self).__init__(n_states, n_actions, seed)

        self.n_steps = 0
        self.state = None
        self.max_steps = max_steps
        self.pi = self.pi_init_value(n_states, pi)

    def each_state(self, f: Callable[[int], Any]):
        return [f(state) for state in range(self.n_states)]

    def each_action(self, f: Callable[[int], Any]):
        return [f(action) for action in range(self.n_actions)]

    @staticmethod
    def pi_init_value(n_states, pi):
        if pi is None:
            return np.full(n_states, 1. / n_states)
        return pi

    def reset(self):
        self.n_steps = 0
        # print(f'n_states={self.n_states}, pi={self.pi}')
        self.state = self.random.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception(f'Invalid action({action}).')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    @abstractmethod
    def render(self, policy=None, value=None):
        raise NotImplementedError()
