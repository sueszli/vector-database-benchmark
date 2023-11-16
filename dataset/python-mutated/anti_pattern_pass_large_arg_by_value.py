import ray
import numpy as np
ray.init()

@ray.remote
def func(large_arg, i):
    if False:
        return 10
    return len(large_arg) + i
large_arg = np.zeros(1024 * 1024)
outputs = ray.get([func.remote(large_arg, i) for i in range(10)])
large_arg_ref = ray.put(large_arg)
outputs = ray.get([func.remote(large_arg_ref, i) for i in range(10)])