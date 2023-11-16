from loguru import logger
import sys

@logger.catch
def main():
    if False:
        for i in range(10):
            print('nop')
    errors = [ValueError(i) for i in range(100)]
    raise ExceptionGroup('group', errors)
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)
logger.add(sys.stderr, format='', diagnose=True, backtrace=True, colorize=True)
main()