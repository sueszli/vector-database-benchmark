import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=False, backtrace=True, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        print('Hello World!')
    b()

def b():
    if False:
        return 10
    c()

def c():
    if False:
        return 10
    d()

def d():
    if False:
        print('Hello World!')
    e()

def e():
    if False:
        for i in range(10):
            print('nop')
    f()

def f():
    if False:
        return 10
    g()

def g():
    if False:
        for i in range(10):
            print('nop')
    h()

def h():
    if False:
        i = 10
        return i + 15
    i()

def i():
    if False:
        return 10
    j(1, 0)

def j(a, b):
    if False:
        return 10
    a / b
sys.tracebacklimit = -1
try:
    a()
except ZeroDivisionError:
    logger.exception('')