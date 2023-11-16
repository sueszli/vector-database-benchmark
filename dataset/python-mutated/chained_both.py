import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def div(x, y):
    if False:
        while True:
            i = 10
    x / y

def cause(x, y):
    if False:
        i = 10
        return i + 15
    try:
        div(x, y)
    except Exception:
        raise ValueError('Division error')

def context(x, y):
    if False:
        print('Hello World!')
    try:
        cause(x, y)
    except Exception as e:
        raise ValueError('Cause error') from e
try:
    context(1, 0)
except ValueError:
    logger.exception('')