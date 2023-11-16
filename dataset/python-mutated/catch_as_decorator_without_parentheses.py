import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)

@logger.catch
def c(a, b=0):
    if False:
        for i in range(10):
            print('nop')
    a / b
c(2)