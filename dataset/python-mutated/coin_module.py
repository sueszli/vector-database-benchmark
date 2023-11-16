"""A custom Python package example.

This package requires that your environment has the scipy PyPI package
installed. """
import numpy as np
from scipy import special

def flip_coin():
    if False:
        for i in range(10):
            print('nop')
    'Return "Heads" or "Tails" depending on a calculation.'
    rand_array = 10 * np.random.random((2, 2)) - 5
    avg = rand_array.mean()
    ndtr = special.ndtr(avg)
    return 'Heads' if ndtr > 0.5 else 'Tails'