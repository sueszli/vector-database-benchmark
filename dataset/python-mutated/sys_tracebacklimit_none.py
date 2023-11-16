import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=False, backtrace=True, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        while True:
            i = 10
    b()

def b():
    if False:
        i = 10
        return i + 15
    c()

def c():
    if False:
        for i in range(10):
            print('nop')
    d()

def d():
    if False:
        print('Hello World!')
    e()

def e():
    if False:
        i = 10
        return i + 15
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
        for i in range(10):
            print('nop')
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
sys.tracebacklimit = None
try:
    a()
except ZeroDivisionError:
    logger.exception('')