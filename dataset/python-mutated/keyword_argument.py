import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def f(x):
    if False:
        for i in range(10):
            print('nop')
    return 1 / x
y = 0
with logger.catch():
    f(x=y)
x = 0
with logger.catch():
    f(x=x)