"""Dependency Injector Factory providers benchmark."""
import time
from dependency_injector import providers
N = 1000000

class A(object):
    pass

class B(object):
    pass

class C(object):
    pass

class Test(object):

    def __init__(self, a, b, c):
        if False:
            return 10
        self.a = a
        self.b = b
        self.c = c
test_factory_provider = providers.Factory(Test, a=providers.Factory(A), b=providers.Factory(B), c=providers.Factory(C))
start = time.time()
for _ in range(1, N):
    test_factory_provider()
finish = time.time()
print(finish - start)

def test_simple_factory_provider():
    if False:
        print('Hello World!')
    return Test(a=A(), b=B(), c=C())
start = time.time()
for _ in range(1, N):
    test_simple_factory_provider()
finish = time.time()
print(finish - start)