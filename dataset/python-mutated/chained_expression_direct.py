import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

@logger.catch()
def a_decorated():
    if False:
        i = 10
        return i + 15
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError('NOK')

def a_not_decorated():
    if False:
        i = 10
        return i + 15
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError('NOK')

def b_decorator():
    if False:
        i = 10
        return i + 15
    a_decorated()

def b_context_manager():
    if False:
        while True:
            i = 10
    with logger.catch():
        a_not_decorated()

def b_explicit():
    if False:
        while True:
            i = 10
    try:
        a_not_decorated()
    except ValueError:
        logger.exception('')
b_decorator()
b_context_manager()
b_explicit()