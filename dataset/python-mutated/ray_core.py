import time
import ray

@ray.remote
def my_function():
    if False:
        i = 10
        return i + 15
    return 1
obj_ref = my_function.remote()
assert ray.get(obj_ref) == 1

@ray.remote
def slow_function():
    if False:
        return 10
    time.sleep(10)
    return 1
results = []
for _ in range(4):
    results.append(slow_function.remote())
ray.get(results)