import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def a(x, y):
    if False:
        return 10
    x / y

@logger.catch
def b_decorated():
    if False:
        i = 10
        return i + 15
    try:
        a(1, 0)
    except ZeroDivisionError as e:
        raise ValueError('NOK') from e

def b_not_decorated():
    if False:
        i = 10
        return i + 15
    try:
        a(1, 0)
    except ZeroDivisionError as e:
        raise ValueError('NOK') from e

def c_decorator():
    if False:
        while True:
            i = 10
    b_decorated()

def c_context_manager():
    if False:
        for i in range(10):
            print('nop')
    with logger.catch():
        b_not_decorated()

def c_explicit():
    if False:
        return 10
    try:
        b_not_decorated()
    except ValueError:
        logger.exception('')
c_decorator()
c_context_manager()
c_explicit()