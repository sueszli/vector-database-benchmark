import ray

@ray.remote
def f():
    if False:
        for i in range(10):
            print('nop')
    raise Exception('the real error')

@ray.remote
def g(x):
    if False:
        while True:
            i = 10
    return
try:
    ray.get(f.remote())
except ray.exceptions.RayTaskError as e:
    print(e)
try:
    ray.get(g.remote(f.remote()))
except ray.exceptions.RayTaskError as e:
    print(e)