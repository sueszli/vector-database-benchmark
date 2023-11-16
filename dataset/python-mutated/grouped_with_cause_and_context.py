import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=False)

def a():
    if False:
        for i in range(10):
            print('nop')
    1 / 0

@logger.catch
def main():
    if False:
        print('Hello World!')
    try:
        try:
            a()
        except Exception as err:
            raise ValueError('ContextError') from err
    except Exception as err:
        from_context = err
    try:
        try:
            a()
        except Exception as err:
            raise ValueError('CauseError')
    except Exception as err:
        from_cause = err
    try:
        a()
    except Exception as err:
        try:
            raise ValueError('Error') from err
        except Exception:
            raise ExceptionGroup('from_context', [from_context, from_cause])
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=True)
main()