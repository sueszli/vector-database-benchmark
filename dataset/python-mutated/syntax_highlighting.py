import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def a():
    if False:
        i = 10
        return i + 15
    1 / 0 + 1 * 0 - 1 % 0 // 1 ** 0 @ 1

def b():
    if False:
        while True:
            i = 10
    a() or False == None != True

def c():
    if False:
        while True:
            i = 10
    (1, 2.5, 3.0, 0.4, 'str', 'rrr', b'binary', b())

def d():
    if False:
        print('Hello World!')
    (min(range(1, 10)), list(), dict(), c(), ...)

def e(x):
    if False:
        i = 10
        return i + 15
    (x in [1], x in (1,), x in {1}, x in {1: 1}, d())
try:
    e(0)
except ZeroDivisionError:
    logger.exception('')