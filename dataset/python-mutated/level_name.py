import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='{level.name} | {level.no}', diagnose=False, backtrace=False, colorize=False)

def a():
    if False:
        return 10
    with logger.catch(level='DEBUG'):
        1 / 0
a()