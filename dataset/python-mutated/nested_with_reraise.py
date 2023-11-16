import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=False, backtrace=True, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

@logger.catch(reraise=True)
def foo(a, b):
    if False:
        i = 10
        return i + 15
    a / b

@logger.catch
def bar(x, y):
    if False:
        return 10
    try:
        f = foo(x, y)
    except Exception as e:
        raise ValueError from e

def baz():
    if False:
        print('Hello World!')
    bar(1, 0)
if __name__ == '__main__':
    baz()