import ray
ray.init()

@ray.remote
def func(src, inc=1):
    if False:
        while True:
            i = 10
    return src + inc
a_ref = func.bind(1, inc=2)
assert ray.get(a_ref.execute()) == 3
b_ref = func.bind(a_ref, inc=3)
assert ray.get(b_ref.execute()) == 6
c_ref = func.bind(b_ref, inc=a_ref)
assert ray.get(c_ref.execute()) == 9
ray.shutdown()
import ray
ray.init()

@ray.remote
class Actor:

    def __init__(self, init_value):
        if False:
            i = 10
            return i + 15
        self.i = init_value

    def inc(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.i += x

    def get(self):
        if False:
            while True:
                i = 10
        return self.i
a1 = Actor.bind(10)
val = a1.get.bind()
assert ray.get(val.execute()) == 10

@ray.remote
def combine(x, y):
    if False:
        print('Hello World!')
    return x + y
a2 = Actor.bind(10)
a1.inc.bind(2)
a1.inc.bind(4)
a2.inc.bind(6)
dag = combine.bind(a1.get.bind(), a2.get.bind())
assert ray.get(dag.execute()) == 32
ray.shutdown()
import ray
ray.init()
from ray.dag.input_node import InputNode

@ray.remote
def a(user_input):
    if False:
        for i in range(10):
            print('nop')
    return user_input * 2

@ray.remote
def b(user_input):
    if False:
        while True:
            i = 10
    return user_input + 1

@ray.remote
def c(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x + y
with InputNode() as dag_input:
    a_ref = a.bind(dag_input)
    b_ref = b.bind(dag_input)
    dag = c.bind(a_ref, b_ref)
assert ray.get(dag.execute(2)) == 7
assert ray.get(dag.execute(3)) == 10