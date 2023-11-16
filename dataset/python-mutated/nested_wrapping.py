import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def f(i):
    if False:
        print('Hello World!')
    1 / i

@logger.catch
@logger.catch()
def a(x):
    if False:
        for i in range(10):
            print('nop')
    f(x)
a(0)
with logger.catch():
    with logger.catch():
        f(0)
try:
    try:
        f(0)
    except ZeroDivisionError:
        logger.exception('')
except Exception:
    logger.exception('')