"""
Topic: 可调整属性的装饰器
Desc : 
"""
from functools import wraps, partial
import logging
import time

def attach_wrapper(obj, func=None):
    if False:
        print('Hello World!')
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)

def logged(level, name=None, message=None):
    if False:
        while True:
            i = 10
    "\n    Add logging to a function. level is the logging\n    level, name is the logger name, and message is the\n    log message. If name and message aren't specified,\n    they default to the function's module and name.\n    "

    def decorate(func):
        if False:
            while True:
                i = 10
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            log.log(level, logmsg)
            return func(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_level(newlevel):
            if False:
                while True:
                    i = 10
            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper)
        def set_message(newmsg):
            if False:
                return 10
            nonlocal logmsg
            logmsg = newmsg
        return wrapper
    return decorate

@logged(logging.DEBUG)
def add(x, y):
    if False:
        i = 10
        return i + 15
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    if False:
        i = 10
        return i + 15
    print('Spam!')
logging.basicConfig(level=logging.DEBUG)
add(2, 3)
add.set_message('Add called')
add(2, 3)
add.set_level(logging.WARNING)
add(2, 3)
print('---------------------')
spam()
spam.set_message('Add called')
spam()
spam.set_level(logging.WARNING)
spam()

def timethis(func):
    if False:
        while True:
            i = 10
    '\n    Decorator that reports the execution time.\n    '

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper

@logged(logging.DEBUG)
@timethis
def countdown(n):
    if False:
        for i in range(10):
            print('nop')
    while n > 0:
        n -= 1
countdown(10000000)
countdown.set_level(logging.WARNING)
countdown.set_message('Counting down to zero')
countdown(10000000)

@timethis
@logged(logging.DEBUG)
def countdown1(n):
    if False:
        print('Hello World!')
    while n > 0:
        n -= 1
print('****************************************')
countdown1(10000000)
countdown1.set_level(logging.WARNING)
countdown1.set_message('Counting down to zero')
countdown1(10000000)