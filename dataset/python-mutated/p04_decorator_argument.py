"""
Topic: 带有参数的装饰器
Desc : 
"""
from functools import wraps
import logging

def logged(level, name=None, message=None):
    if False:
        return 10
    "\n    Add logging to a function. level is the logging\n    level, name is the logger name, and message is the\n    log message. If name and message aren't specified,\n    they default to the function's module and name.\n    "

    def decorate(func):
        if False:
            for i in range(10):
                print('nop')
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
        return wrapper
    return decorate

@logged(logging.DEBUG)
def add(x, y):
    if False:
        print('Hello World!')
    return x + y

@logged(logging.CRITICAL, 'example')
def spam():
    if False:
        while True:
            i = 10
    print('Spam!')