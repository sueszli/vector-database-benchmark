import sys
import _init
from somelib import callme, divide
from loguru import logger

def test(*, backtrace, colorize, diagnose):
    if False:
        print('Hello World!')
    logger.remove()
    logger.add(sys.stderr, format='', colorize=colorize, backtrace=backtrace, diagnose=diagnose)

    @logger.catch
    def callback():
        if False:
            i = 10
            return i + 15
        (a, b) = (1, 0)
        a / b
    callme(callback)
test(backtrace=True, colorize=True, diagnose=True)
test(backtrace=False, colorize=True, diagnose=True)
test(backtrace=True, colorize=True, diagnose=False)
test(backtrace=False, colorize=True, diagnose=False)
test(backtrace=False, colorize=False, diagnose=False)