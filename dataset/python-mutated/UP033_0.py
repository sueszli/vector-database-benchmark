import functools

@functools.lru_cache(maxsize=None)
def fixme():
    if False:
        print('Hello World!')
    pass

@other_decorator
@functools.lru_cache(maxsize=None)
def fixme():
    if False:
        while True:
            i = 10
    pass

@functools.lru_cache(maxsize=None)
@other_decorator
def fixme():
    if False:
        i = 10
        return i + 15
    pass

@functools.lru_cache()
def ok():
    if False:
        while True:
            i = 10
    pass

@functools.lru_cache(maxsize=64)
def ok():
    if False:
        return 10
    pass

def user_func():
    if False:
        print('Hello World!')
    pass

@functools.lru_cache(user_func)
def ok():
    if False:
        print('Hello World!')
    pass

def lru_cache(maxsize=None):
    if False:
        while True:
            i = 10
    pass

@lru_cache(maxsize=None)
def ok():
    if False:
        for i in range(10):
            print('nop')
    pass