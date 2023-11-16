from ray.util.client import ray
from typing import Tuple
ray.connect('localhost:50051')

@ray.remote
class HelloActor:

    def __init__(self):
        if False:
            return 10
        self.count = 0

    def say_hello(self, whom: str) -> Tuple[str, int]:
        if False:
            return 10
        self.count += 1
        return ('Hello ' + whom, self.count)
actor = HelloActor.remote()
(s, count) = ray.get(actor.say_hello.remote('you'))
print(s, count)
assert s == 'Hello you'
assert count == 1
(s, count) = ray.get(actor.say_hello.remote('world'))
print(s, count)
assert s == 'Hello world'
assert count == 2

@ray.remote
def plus2(x):
    if False:
        return 10
    return x + 2

@ray.remote
def fact(x):
    if False:
        i = 10
        return i + 15
    print(x, type(fact))
    if x <= 0:
        return 1
    return ray.get(fact.remote(x - 1)) * x

@ray.remote
def get_nodes():
    if False:
        i = 10
        return i + 15
    return ray.nodes()
print('Cluster nodes', ray.get(get_nodes.remote()))
print(ray.nodes())
objectref = ray.put('hello world')
print(objectref)
print(ray.get(objectref))
ref2 = plus2.remote(234)
print(ref2)
print(ray.get(ref2))
ref3 = fact.remote(20)
print(ref3)
print(ray.get(ref3))
ref4 = fact.remote(5)
print(ray.get(ref4))
ref5 = fact.remote(10)
print([ref2, ref3, ref4, ref5])
res = ray.wait([ref5, ref2, ref3, ref4], num_returns=3)
print(res)
assert [ref2, ref3, ref4] == res[0]
assert [ref5] == res[1]
res = ray.wait([ref2, ref3, ref4, ref5], num_returns=4)
print(res)
assert [ref2, ref3, ref4, ref5] == res[0]
assert [] == res[1]