import ray
ray.init()

@ray.remote
def f(a: int) -> str:
    if False:
        while True:
            i = 10
    return 'a = {}'.format(a + 1)

@ray.remote
def g(s: str) -> str:
    if False:
        return 10
    return s + ' world'

@ray.remote
def h(a: str, b: int) -> str:
    if False:
        i = 10
        return i + 15
    return a
print(f.remote(1))
object_ref_str = f.remote(1)
print(g.remote(object_ref_str))
print(h.remote(object_ref_str, 100))
xy = ray.get(object_ref_str) + 'y'