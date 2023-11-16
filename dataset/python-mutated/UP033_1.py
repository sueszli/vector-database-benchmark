from functools import lru_cache

@lru_cache(maxsize=None)
def fixme():
    if False:
        while True:
            i = 10
    pass

@other_decorator
@lru_cache(maxsize=None)
def fixme():
    if False:
        i = 10
        return i + 15
    pass

@lru_cache(maxsize=None)
@other_decorator
def fixme():
    if False:
        return 10
    pass

@lru_cache()
def ok():
    if False:
        for i in range(10):
            print('nop')
    pass

@lru_cache(maxsize=64)
def ok():
    if False:
        while True:
            i = 10
    pass

def user_func():
    if False:
        for i in range(10):
            print('nop')
    pass

@lru_cache(user_func)
def ok():
    if False:
        return 10
    pass

@lru_cache(user_func, maxsize=None)
def ok():
    if False:
        while True:
            i = 10
    pass

def lru_cache(maxsize=None):
    if False:
        return 10
    pass

@lru_cache(maxsize=None)
def ok():
    if False:
        while True:
            i = 10
    pass