import ray
import numpy as np
ray.init()

@ray.remote
def generate_rollout():
    if False:
        for i in range(10):
            print('nop')
    return np.ones((10000, 10000))

@ray.remote
def reduce(rollout):
    if False:
        return 10
    return np.sum(rollout)
rollout = ray.get(generate_rollout.remote())
reduced = ray.get(reduce.remote(rollout))
assert reduced == 100000000
rollout_obj_ref = generate_rollout.remote()
reduced = ray.get(reduce.remote(rollout_obj_ref))
assert reduced == 100000000