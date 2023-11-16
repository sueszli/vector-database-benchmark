import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def foo():
    if False:
        return 10
    raise ValueError('')

def bar():
    if False:
        while True:
            i = 10
    foo()
try:
    bar()
except ValueError:
    logger.exception('')