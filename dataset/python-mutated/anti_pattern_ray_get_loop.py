import ray
ray.init()

@ray.remote
def f(i):
    if False:
        while True:
            i = 10
    return i
sequential_returns = []
for i in range(100):
    sequential_returns.append(ray.get(f.remote(i)))
refs = []
for i in range(100):
    refs.append(f.remote(i))
parallel_returns = ray.get(refs)
assert sequential_returns == parallel_returns