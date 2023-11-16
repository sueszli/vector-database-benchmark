"""
Topic: 将装饰器定义成类
Desc : 
"""
import types
from functools import wraps

class Profiled:

    def __init__(self, func):
        if False:
            print('Hello World!')
        wraps(func)(self)
        self.ncalls = 0

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.ncalls += 1
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        if False:
            return 10
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)

@Profiled
def add(x, y):
    if False:
        return 10
    return x + y

class Spam:

    @Profiled
    def bar(self, x):
        if False:
            print('Hello World!')
        print(self, x)
Spam().bar(3)
import types
from functools import wraps

def profiled(func):
    if False:
        for i in range(10):
            print('nop')
    ncalls = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        nonlocal ncalls
        ncalls += 1
        return func(*args, **kwargs)
    wrapper.ncalls = lambda : ncalls
    return wrapper

@profiled
def add(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x + y