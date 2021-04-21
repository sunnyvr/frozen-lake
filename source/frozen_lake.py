import numpy as np

from config import small_lake, _printoptions
from environment import Environment

N_ACTIONS = 4
CAN_SLIP_TO_SAME_POSITION = False


def action_direction(action_id):
    """Actions are wasd (up, left, down, right)"""
    directions = ([(-1, 0), (0, -1), (1, 0), (0, 1)])
    return np.array(directions[action_id])


def is_next_to(first, second):
    diff_squared = (first - second) ** 2
    return diff_squared.sum() == 1


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None, print_options=None):
        super().__init__(self.calculate_state_count(lake),
                         n_actions=N_ACTIONS, max_steps=max_steps, pi=None, seed=seed)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        self.print_options = print_options

        self.pi = np.zeros(self.n_states, dtype=float)
        self.pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = self.n_states - 1

    @property
    def shape(self):
        return self.lake.shape

    @property
    def terminal_state(self):
        return self.absorbing_state

    @property
    def n_rows(self):
        return self.shape[0]

    @property
    def n_cols(self):
        return self.shape[1]

    def state_coords(self, state):
        if state == self.absorbing_state:
            raise RuntimeError()

        row = state // self.n_cols
        col = state % self.n_cols
        return np.array([row, col])

    def action_coords(self, state, action):
        position = self.state_coords(state)
        direction = action_direction(action)
        target = position + direction
        return target

    @staticmethod
    def calculate_state_count(lake):
        # To make the constructor a little tidier.
        return np.asarray(lake).size + 1

    @staticmethod
    def create(lake, slip, max_steps, seed=None, print_options=None):
        lake = np.array(lake)
        return FrozenLake(lake, slip, max_steps, seed, print_options)

    def apply_friction(self, action):
        if self.random.rand() >= self.slip:
            return action
        slip_action = self.random.randint(self.n_actions)
        return slip_action

    def step(self, action):
        last_state = self.state
        action = self.apply_friction(action)
        state, reward, done = Environment.step(self, action)
        if self.is_goal_or_hole(last_state):
            state = self.absorbing_state
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def is_goal_or_hole(self, state):
        return state < len(self.lake_flat) and self.lake_flat[state] in "$#"

    def is_hole(self, state):
        return state < len(self.lake_flat) and self.lake_flat[state] == '#'

    def p(self, next_state, state, action):
        try:
            if self.is_hole(state) or self.is_absorbing_state(state) or self.is_absorbing_state(next_state):
                return 0

            target_position = self.state_coords(next_state)
            start_position = self.state_coords(state)
            action_position = self.action_coords(state, action)

            is_adjacent = is_next_to(target_position, start_position)
            is_target = np.array_equal(target_position, action_position)

            return 1 if (is_adjacent and is_target) else 0
        except:
            print(f'### Error: something has gone terribly wrong in FrozenLake.p()')
            print(f'           state={state}, next_state={next_state}, action={action}')
            return 0

    def are_states_adjacent(self, first_state, second_state):
        first_position = self.state_coords(first_state)
        second_position = self.state_coords(second_state)
        return is_next_to(first_position, second_position)

    def state_is_goal(self, state):
        if state == self.absorbing_state:
            return False
        return self.lake_flat[state] == '$'

    def is_absorbing_state(self, state):
        return state == self.absorbing_state

    def r(self, next_state, state, action):
        try:
            if self.is_absorbing_state(state) or self.is_absorbing_state(next_state):
                return 0

            target_position = self.state_coords(next_state)
            start_position = self.state_coords(state)
            action_position = self.action_coords(state, action)
            is_adjacent = is_next_to(start_position, action_position)
            is_at_target = np.array_equal(action_position, target_position)
            is_rewarded = is_adjacent and self.state_is_goal(state) and is_at_target
            return 1 if is_rewarded else 0
        except:
            print(f'### Error: something has gone wrong in FrozenLake.r()')
            print(f'           state={state}, next_state={next_state}, action={action}')
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            actions = [*"^<_>"]

            print('Lake: ')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            print('Value:')
            self.print(value[:-1].reshape(self.lake.shape))

    def print(self, *args, **kwargs):
        if self.print_options:
            with self.print_options(precision=3, suppress=True):
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)


def play(env):
    actions = [*"wasd"]

    _state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid Action')

        _state, r, done = env.step(actions.index(c))

        env.render()
        print(f'Reward: {r}.')


def main():
    lake = small_lake()
    print(f'Lake = {lake}')
    print_options = _printoptions
    seed = 0
    env = FrozenLake.create(lake, slip=0.1, max_steps=16, seed=seed, print_options=print_options)
    play(env)


if __name__ == "__main__":
    main()
