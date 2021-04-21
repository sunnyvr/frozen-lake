from typing import Callable, Any

import numpy as np

from environment import Environment
from logger import Logger


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    iteration = 1
    for i in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            value[state] = sum([env.p(next_state, state, policy[state]) * (
                    env.r(next_state, state, policy[state]) + gamma * value[next_state]) for next_state in
                                range(env.n_states)])
            delta = max(delta, abs(v - value[state]))

        if delta < theta:
            break
        iteration += 1
    return value


def calc_action_value(env: Environment, state, action, value, gamma):
    def calculate_reward(next_state):
        p = env.p(next_state, state, action)
        r = env.r(next_state, state, action)
        v = value[next_state]
        return p * (r + gamma * v)

    return np.sum(env.each_state(calculate_reward))


def best_action_estimate(env: Environment, state, value, gamma):
    return np.max(env.each_action(lambda action: calc_action_value(env, state, action, value, gamma)))


def optimal_action_for_state(env: Environment, state, value, gamma):
    return np.argmax([calc_action_value(env, state, action, value, gamma) for action in range(env.n_actions)])


def policy_improvement(env: Environment, value, gamma, state=None):
    if state is not None:
        return optimal_action_for_state(env, state, value, gamma)

    return np.array(env.each_state(lambda s: optimal_action_for_state(env, s, value, gamma)), dtype=int)


def converge_policy_improvement(env, values, gamma, old_policy):
    policy = policy_improvement(env, values, gamma)
    has_converged = np.all(policy == old_policy)
    return policy, has_converged


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    iterations = 0
    has_converged = False

    values = None

    while not has_converged:
        values = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, has_converged = converge_policy_improvement(env, values, gamma, policy)
        iterations += 1

    print(f'Iterations {str(iterations)}')
    return policy, values


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    log = Logger.with_max_index(max_iterations, 0.05)

    for i in range(max_iterations):
        log.set_index(i)
        delta = 0

        for state in range(env.n_states):
            v = value[state]
            value[state] = best_action_estimate(env, state, value, gamma)
            delta = max(delta, np.abs(value[state] - v))

        if delta < theta:
            print('value converged at iteration', i)
            break

    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
        policy[state] = policy_improvement(env, value, gamma, state)

    # policy = None
    print(policy, value)
    return policy, value
