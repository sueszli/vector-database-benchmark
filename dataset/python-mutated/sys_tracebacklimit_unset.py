import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=False, backtrace=True, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        return 10
    b()

def b():
    if False:
        i = 10
        return i + 15
    c()

def c():
    if False:
        print('Hello World!')
    d()

def d():
    if False:
        while True:
            i = 10
    e()

def e():
    if False:
        print('Hello World!')
    f()

def f():
    if False:
        i = 10
        return i + 15
    g()

def g():
    if False:
        i = 10
        return i + 15
    h()

def h():
    if False:
        while True:
            i = 10
    i()

def i():
    if False:
        for i in range(10):
            print('nop')
    j(1, 0)

def j(a, b):
    if False:
        print('Hello World!')
    a / b
try:
    del sys.tracebacklimit
except AttributeError:
    pass
try:
    a()
except ZeroDivisionError:
    logger.exception('')