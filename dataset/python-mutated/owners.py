import ray
import numpy as np

@ray.remote
def large_array():
    if False:
        for i in range(10):
            print('nop')
    return np.zeros(int(100000.0))
x = ray.put(1)
y = large_array.remote()