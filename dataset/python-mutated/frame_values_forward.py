import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)
k = 2

def a(n):
    if False:
        return 10
    1 / n

def b(n):
    if False:
        return 10
    a(n - 1)

@logger.catch
def c(n):
    if False:
        print('Hello World!')
    b(n - 1)
c(k)