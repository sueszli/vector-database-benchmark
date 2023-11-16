import sys
from loguru import logger

def test(diagnose, backtrace):
    if False:
        return 10
    logger.remove()
    logger.add(sys.stderr, format='', diagnose=diagnose, backtrace=backtrace, colorize=True)
    try:
        assert False
    except AssertionError:
        (type_, value, _) = sys.exc_info()
        logger.opt(exception=(type_, value, None)).error('')
test(False, False)
test(True, False)
test(False, True)
test(True, True)