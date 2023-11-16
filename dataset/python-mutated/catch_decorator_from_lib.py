import sys
import _init
from somelib import callme, divide
from loguru import logger

def test(*, backtrace, colorize, diagnose):
    if False:
        for i in range(10):
            print('nop')
    logger.remove()
    logger.add(sys.stderr, format='', colorize=colorize, backtrace=backtrace, diagnose=diagnose)

    @logger.catch
    def callback():
        if False:
            while True:
                i = 10
        divide(1, 0)
    callme(callback)
test(backtrace=True, colorize=True, diagnose=True)
test(backtrace=False, colorize=True, diagnose=True)
test(backtrace=True, colorize=True, diagnose=False)
test(backtrace=False, colorize=True, diagnose=False)
test(backtrace=False, colorize=False, diagnose=False)