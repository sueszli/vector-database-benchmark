import ray
import numpy as np

@ray.remote
def f(arr):
    if False:
        while True:
            i = 10
    arr[0] = 1
try:
    ray.get(f.remote(np.zeros(100)))
except ray.exceptions.RayTaskError as e:
    print(e)