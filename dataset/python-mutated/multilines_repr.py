import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

class A:

    def __repr__(self):
        if False:
            return 10
        return '[[1, 2, 3]\n [4, 5, 6]\n [7, 8, 9]]'

def multiline():
    if False:
        print('Hello World!')
    a = b = A()
    a + b
try:
    multiline()
except TypeError:
    logger.exception('')