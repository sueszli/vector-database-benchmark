import ray
ray.init()

@ray.remote
def f(a: int) -> str:
    if False:
        return 10
    return 'a = {}'.format(a + 1)

@ray.remote
def g(s: str) -> str:
    if False:
        print('Hello World!')
    return s + ' world'

@ray.remote
def h(a: str, b: int) -> str:
    if False:
        i = 10
        return i + 15
    return a
a = h.remote(1, 1)
b = f.remote('hello')
c = f.remote(1, 1)
d = f.remote(1) + 1
ref_to_str = f.remote(1)
unwrapped_str = ray.get(ref_to_str)
unwrapped_str + 100
f.remote(ref_to_str)