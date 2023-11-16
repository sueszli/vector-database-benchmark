"""
Topic: 带可选参数的装饰器
Desc : 
"""
from functools import wraps, partial
import logging

def logged(func=None, *, level=logging.DEBUG, name=None, message=None):
    if False:
        for i in range(10):
            print('nop')
    if func is None:
        return partial(logged, level=level, name=name, message=message)
    logname = name if name else func.__module__
    log = logging.getLogger(logname)
    logmsg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        log.log(level, logmsg)
        return func(*args, **kwargs)
    return wrapper

@logged
def add(x, y):
    if False:
        while True:
            i = 10
    return x + y

@logged(level=logging.CRITICAL, name='example')
def spam():
    if False:
        return 10
    print('Spam!')
spam()

def aa(kk=None, *, a=1, b=2, c=3):
    if False:
        i = 10
        return i + 15
    print(kk, a, b, c)
bbb = partial(aa, a=1, b=2, c=3)
bbb('333')