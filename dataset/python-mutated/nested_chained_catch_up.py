import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=False, backtrace=False, diagnose=False)
logger.add(sys.stderr, format='', colorize=False, backtrace=True, diagnose=False)

def foo():
    if False:
        return 10
    bar()

@logger.catch(ValueError)
def bar():
    if False:
        i = 10
        return i + 15
    1 / 0

@logger.catch
def main():
    if False:
        while True:
            i = 10
    try:
        foo()
    except Exception as e:
        raise ZeroDivisionError from e
main()