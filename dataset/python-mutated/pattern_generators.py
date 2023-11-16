import sys
if len(sys.argv) == 1:
    sys.argv.append('5')
import sys
import ray
import numpy as np

@ray.remote
def large_values(num_returns):
    if False:
        return 10
    return [np.random.randint(np.iinfo(np.int8).max, size=(100000000, 1), dtype=np.int8) for _ in range(num_returns)]

@ray.remote
def large_values_generator(num_returns):
    if False:
        i = 10
        return i + 15
    for i in range(num_returns):
        yield np.random.randint(np.iinfo(np.int8).max, size=(100000000, 1), dtype=np.int8)
        print(f'yielded return value {i}')
num_returns = int(sys.argv[1])
print('Using normal functions...')
try:
    ray.get(large_values.options(num_returns=num_returns, max_retries=0).remote(num_returns)[0])
except ray.exceptions.WorkerCrashedError:
    print('Worker failed with normal function')
print('Using generators...')
ray.get(large_values_generator.options(num_returns=num_returns, max_retries=0).remote(num_returns)[0])
print('Success!')