import ray
import numpy as np
ray.init()
large_object = np.zeros(10 * 1024 * 1024)

@ray.remote
def f1():
    if False:
        return 10
    return len(large_object)
ray.get(f1.remote())
large_object_ref = ray.put(np.zeros(10 * 1024 * 1024))

@ray.remote
def f2(large_object):
    if False:
        return 10
    return len(large_object)
ray.get(f2.remote(large_object_ref))
large_object_creator = lambda : np.zeros(10 * 1024 * 1024)

@ray.remote
def f3():
    if False:
        i = 10
        return i + 15
    large_object = large_object_creator()
    return len(large_object)
ray.get(f3.remote())