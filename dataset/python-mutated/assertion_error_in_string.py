import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def foo(abc, xyz):
    if False:
        for i in range(10):
            print('nop')
    exec('assert abc > 10 and xyz == 60')
try:
    foo(9, 55)
except AssertionError:
    logger.exception('')