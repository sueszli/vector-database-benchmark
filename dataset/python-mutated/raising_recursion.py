import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

@logger.catch
def a(n):
    if False:
        i = 10
        return i + 15
    if n:
        a(n - 1)
    n / 0

def b(n):
    if False:
        return 10
    with logger.catch():
        if n:
            b(n - 1)
        n / 0

def c(n):
    if False:
        print('Hello World!')
    try:
        if n:
            c(n - 1)
        n / 0
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