import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        while True:
            i = 10
    1 / 0

def b():
    if False:
        while True:
            i = 10
    raise ValueError('Error')

@logger.catch
def main():
    if False:
        i = 10
        return i + 15
    try:
        a()
    except Exception as err:
        error_1 = err
    try:
        b()
    except Exception as err:
        error_2 = err
    try:
        try:
            raise ExceptionGroup('group_1', [error_1, error_2])
        except Exception as err:
            raise ExceptionGroup('group_2', [error_2, error_1]) from err
    except Exception as err:
        raise ExceptionGroup('group_3', [err])
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=True)
main()