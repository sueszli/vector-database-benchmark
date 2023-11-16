import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

@logger.catch()
def a(n):
    if False:
        return 10
    if n:
        a(n - 1)
    1 / n

def b(n):
    if False:
        while True:
            i = 10
    if n:
        with logger.catch():
            b(n - 1)
    1 / n

def c(n):
    if False:
        for i in range(10):
            print('nop')
    if n:
        try:
            c(n - 1)
        except ZeroDivisionError:
            logger.exception('')
    1 / n
a(1)
a(2)
a(3)
b(1)
b(2)
b(3)
c(1)
c(2)
c(3)