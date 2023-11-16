import sys
import _init
from somelib import divide
from loguru import logger

def test(*, backtrace, colorize, diagnose):
    if False:
        while True:
            i = 10
    logger.remove()
    logger.add(sys.stderr, format='', colorize=colorize, backtrace=backtrace, diagnose=diagnose)

    @logger.catch
    def foo():
        if False:
            while True:
                i = 10
        divide(1, 0)
    foo()
test(backtrace=True, colorize=True, diagnose=True)
test(backtrace=False, colorize=True, diagnose=True)
test(backtrace=True, colorize=True, diagnose=False)
test(backtrace=False, colorize=True, diagnose=False)
test(backtrace=False, colorize=False, diagnose=False)