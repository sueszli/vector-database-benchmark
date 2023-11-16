from prml.nn.array.array import Array
from prml.nn.config import config
import numpy as np

def ones(size):
    if False:
        for i in range(10):
            print('nop')
    return Array(np.ones(size, dtype=config.dtype))