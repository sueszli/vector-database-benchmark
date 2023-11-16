import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)
k = 2

@logger.catch
def a(n):
    if False:
        i = 10
        return i + 15
    1 / n

def b(n):
    if False:
        for i in range(10):
            print('nop')
    a(n - 1)

def c(n):
    if False:
        print('Hello World!')
    b(n - 1)
c(k)