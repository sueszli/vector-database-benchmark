import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='{name} {file.name} {function} {line}', diagnose=False, backtrace=False, colorize=False)

def a():
    if False:
        while True:
            i = 10
    with logger.catch():
        1 / 0
a()