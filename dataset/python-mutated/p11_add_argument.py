"""
Topic: 给被包装函数增加参数
Desc : 
"""
from functools import wraps
import inspect

def optional_debug(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if False:
            return 10
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    return wrapper

@optional_debug
def spam(a, b, c):
    if False:
        print('Hello World!')
    print(a, b, c)
spam(1, 2, 3)
spam(1, 2, 3, debug=True)

def a(x, debug=False):
    if False:
        print('Hello World!')
    if debug:
        print('Calling a')

def b(x, y, z, debug=False):
    if False:
        while True:
            i = 10
    if debug:
        print('Calling b')

def c(x, y, debug=False):
    if False:
        return 10
    if debug:
        print('Calling c')

def optional_debug(func):
    if False:
        for i in range(10):
            print('nop')
    if 'debug' in inspect.getargspec(func).args:
        raise TypeError('debug argument already defined')

    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if False:
            while True:
                i = 10
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    return wrapper

@optional_debug
def a(x):
    if False:
        for i in range(10):
            print('nop')
    pass

@optional_debug
def b(x, y, z):
    if False:
        i = 10
        return i + 15
    pass

@optional_debug
def c(x, y):
    if False:
        while True:
            i = 10
    pass

def optional_debug1(func):
    if False:
        for i in range(10):
            print('nop')
    if 'debug' in inspect.getargspec(func).args:
        raise TypeError('debug argument already defined')

    @wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if False:
            return 10
        if debug:
            print('Calling', func.__name__)
        return func(*args, **kwargs)
    sig = inspect.signature(func)
    parms = list(sig.parameters.values())
    parms.append(inspect.Parameter('debug', inspect.Parameter.KEYWORD_ONLY, default=False))
    wrapper.__signature__ = sig.replace(parameters=parms)
    return wrapper

@optional_debug1
def add(x, y):
    if False:
        return 10
    return x + y
print(inspect.signature(add))