import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def a(x, y):
    if False:
        return 10
    x / y

def b():
    if False:
        while True:
            i = 10
    try:
        a(1, 0)
    except ZeroDivisionError as e:
        raise ValueError('NOK') from e

@logger.catch
def c_decorated():
    if False:
        for i in range(10):
            print('nop')
    b()

def c_not_decorated():
    if False:
        i = 10
        return i + 15
    b()
c_decorated()
with logger.catch():
    c_not_decorated()
try:
    c_not_decorated()
except ValueError:
    logger.exception('')