import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def a_decorator():
    if False:
        print('Hello World!')
    b_decorated()

def a_context_manager():
    if False:
        for i in range(10):
            print('nop')
    with logger.catch():
        b_not_decorated()

def a_explicit():
    if False:
        while True:
            i = 10
    try:
        b_not_decorated()
    except ZeroDivisionError:
        logger.exception('')

@logger.catch()
def b_decorated():
    if False:
        for i in range(10):
            print('nop')
    c()

def b_not_decorated():
    if False:
        print('Hello World!')
    c()

def c():
    if False:
        i = 10
        return i + 15
    1 / 0
a_decorator()
a_context_manager()
a_explicit()