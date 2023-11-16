import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

@logger.catch
def a_decorated():
    if False:
        return 10
    b()

def a_not_decorated():
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
        print('Hello World!')
    1 / 0
a_decorated()
with logger.catch():
    a_not_decorated()
try:
    a_not_decorated()
except ZeroDivisionError:
    logger.exception('')