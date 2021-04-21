import numpy as np
import contextlib


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def small_lake():
    lake = ['&...',
            '.#.#',
            '...#',
            '#..$']
    return [[*each] for each in lake]


def big_frozen_lake():
    lake = ['&.......',
            '........',
            '...#....',
            '.....#..',
            '...#....',
            '.##...#.',
            '.#..#.#.',
            '...#...$']
    return [[*each] for each in lake]
