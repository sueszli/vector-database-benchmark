import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', diagnose=False, backtrace=False, colorize=False)

@logger.catch()
def c(a, b):
    if False:
        return 10
    a / b
c(5, b=0)