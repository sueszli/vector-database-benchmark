import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)
foo = True
bar = False

def func():
    if False:
        i = 10
        return i + 15
    foo = None
    return 1 / 0 + foo + bar + False
try:
    func()
except ZeroDivisionError:
    logger.exception('')