"""
Topic: 包装函数
Desc : 
"""
import time
from functools import wraps

def timethis(func):
    if False:
        i = 10
        return i + 15
    '\n    Decorator that reports the execution time.\n    '

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper

@timethis
def countdown(n):
    if False:
        i = 10
        return i + 15
    '\n    Counts down\n    '
    while n > 0:
        n -= 1
countdown(100000)
countdown(10000000)