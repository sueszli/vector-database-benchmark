import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

@logger.catch()
def a(n):
    if False:
        for i in range(10):
            print('nop')
    1 / n
    a(n - 1)

def b(n):
    if False:
        for i in range(10):
            print('nop')
    1 / n
    with logger.catch():
        b(n - 1)

def c(n):
    if False:
        return 10
    1 / n
    try:
        c(n - 1)
    except ZeroDivisionError:
        logger.exception('')
a(1)
a(2)
a(3)
b(1)
b(2)
b(3)
c(1)
c(2)
c(3)