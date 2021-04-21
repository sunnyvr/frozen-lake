from config import small_lake, _printoptions
from frozen_lake import FrozenLake
from linear_wrapper import LinearWrapper, linear_sarsa, linear_q_learning
from tabular import policy_iteration, value_iteration
from tabular_modelfree_reinforcement import sarsa, q_learning


def main():
    running = {
        'policy': True,
        'value': True,
        'sarsa': True,
        'q': True,
        'sarsa/L': True,
        'q/L': True
    }
    seed = 1
    lake = small_lake()
    env = FrozenLake.create(lake, slip=0.1, max_steps=16, seed=seed, print_options=_printoptions)
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    if running['policy']:
        print('')
        print('## Policy iteration')
        policy, value = policy_iteration(env, gamma, theta, max_iterations)
        env.render(policy, value)

    if running['value']:
        print('')
        print('## Value iteration')
        policy, value = value_iteration(env, gamma, theta, max_iterations)
        env.render(policy, value)

    print('')
    print('# Model free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5

    if running['sarsa']:
        print('\n## Sarsa')
        policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
        env.render(policy, value)

    if running['q']:
        print('\n## Q-learning')
        policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
        env.render(policy, value)

    linear_env = LinearWrapper(env)

    if running['sarsa/L']:
        print('\n## Linear Sarsa')
        parameters = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
        policy, value = linear_env.decode_policy(parameters)
        linear_env.render(policy, value)

    if running['q/L']:
        print('\n## Linear Q-Learning')
        parameters = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
        policy, value = linear_env.decode_policy(parameters)
        linear_env.render(policy, value)


if __name__ == "__main__":
    main()
