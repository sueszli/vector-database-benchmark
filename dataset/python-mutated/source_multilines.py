import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)

def bug_1(n):
    if False:
        for i in range(10):
            print('nop')
    return 'multi-lines\n' + n / 0

def bug_2(a, b, c):
    if False:
        i = 10
        return i + 15
    return 1 / 0 + a + b + c

def bug_3(string):
    if False:
        print('Hello World!')
    return min(10, string, 20 / 0)

def bug_4():
    if False:
        print('Hello World!')
    (a, b) = (1, 0)
    dct = {'foo': 1, 'bar': a / b}
    return dct
string = 'multi-lines\n'
try:
    bug_1(10)
except ZeroDivisionError:
    logger.exception('')
try:
    bug_2(1, string, 3)
except ZeroDivisionError:
    logger.exception('')
try:
    bug_3(string)
except ZeroDivisionError:
    logger.exception('')
try:
    bug_4()
except ZeroDivisionError:
    logger.exception('')