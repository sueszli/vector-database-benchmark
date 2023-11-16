import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def a_decorator():
    if False:
        print('Hello World!')
    b_decorator()

def a_context_manager():
    if False:
        while True:
            i = 10
    b_context_manager()

def a_explicit():
    if False:
        i = 10
        return i + 15
    b_explicit()

def b_decorator():
    if False:
        while True:
            i = 10
    c_decorated()

def b_context_manager():
    if False:
        for i in range(10):
            print('nop')
    with logger.catch():
        c_not_decorated()

def b_explicit():
    if False:
        print('Hello World!')
    try:
        c_not_decorated()
    except ZeroDivisionError:
        logger.exception('')

@logger.catch
def c_decorated():
    if False:
        return 10
    1 / 0

def c_not_decorated():
    if False:
        for i in range(10):
            print('nop')
    1 / 0
a_decorator()
a_context_manager()
a_explicit()