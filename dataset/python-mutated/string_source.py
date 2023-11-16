import sys
from loguru import logger

def test(*, backtrace, colorize, diagnose):
    if False:
        i = 10
        return i + 15
    logger.remove()
    logger.add(sys.stderr, format='', colorize=colorize, backtrace=backtrace, diagnose=diagnose)

    def foo():
        if False:
            return 10
        1 / 0
    try:
        exec('foo()')
    except ZeroDivisionError:
        logger.exception('')
test(backtrace=True, colorize=True, diagnose=True)
test(backtrace=False, colorize=True, diagnose=True)
test(backtrace=True, colorize=True, diagnose=False)
test(backtrace=False, colorize=True, diagnose=False)
test(backtrace=False, colorize=False, diagnose=False)