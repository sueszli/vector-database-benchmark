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
        print('Hello World!')
    c()

def c():
    if False:
        print('Hello World!')
    d()

def d():
    if False:
        for i in range(10):
            print('nop')
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
        return 10
    h()

def h():
    if False:
        while True:
            i = 10
    i()

def i():
    if False:
        return 10
    j(1, 0)

def j(a, b):
    if False:
        return 10
    a / b
sys.tracebacklimit = 5
try:
    a()
except ZeroDivisionError:
    logger.exception('')