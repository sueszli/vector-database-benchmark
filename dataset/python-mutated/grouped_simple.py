import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        return 10
    x = 1
    y = 0
    x / y

def b():
    if False:
        while True:
            i = 10
    a()

def c(f):
    if False:
        print('Hello World!')
    f()

@logger.catch
def main():
    if False:
        return 10
    try:
        c(b)
    except Exception as error_1:
        try:
            c(a)
        except Exception as error_2:
            try:
                a()
            except Exception as error_3:
                raise ExceptionGroup('group', [error_1, error_2, error_3]) from None
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=True)
main()