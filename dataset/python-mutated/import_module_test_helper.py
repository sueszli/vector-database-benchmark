import numpy as np

def cb(x):
    if False:
        print('Hello World!')
    return np.full((10, 100), x.idx_in_epoch)