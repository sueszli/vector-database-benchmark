import ray

@ray.remote
def f():
    if False:
        while True:
            i = 10
    return 1

@ray.remote
def g():
    if False:
        i = 10
        return i + 15
    return [f.remote() for _ in range(4)]

@ray.remote
def h():
    if False:
        print('Hello World!')
    return ray.get([f.remote() for _ in range(4)])

@ray.remote(num_cpus=1, num_gpus=1)
def g():
    if False:
        i = 10
        return i + 15
    return ray.get(f.remote())