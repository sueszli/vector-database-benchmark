import numpy as np

def expit(x: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    return 1 / (1 + np.exp(-x))