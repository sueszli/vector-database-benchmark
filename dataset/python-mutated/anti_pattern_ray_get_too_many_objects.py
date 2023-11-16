import ray
import numpy as np
ray.init()

def process_results(results):
    if False:
        for i in range(10):
            print('nop')
    pass

@ray.remote
def return_big_object():
    if False:
        i = 10
        return i + 15
    return np.zeros(1024 * 10)
NUM_TASKS = 1000
object_refs = [return_big_object.remote() for _ in range(NUM_TASKS)]
results = ray.get(object_refs)
process_results(results)
BATCH_SIZE = 100
while object_refs:
    (ready_object_refs, object_refs) = ray.wait(object_refs, num_returns=BATCH_SIZE)
    results = ray.get(ready_object_refs)
    process_results(results)