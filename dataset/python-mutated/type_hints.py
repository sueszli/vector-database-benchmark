import sys
from typing import TypeVar, Union
from loguru import logger
logger.remove()
logger.add(sys.stderr, format='', colorize=True, backtrace=False, diagnose=True)
T = TypeVar('T')
Name = str

def foo(a: int, b: Union[Name, float], c: 'Name') -> T:
    if False:
        i = 10
        return i + 15
    1 / 0

def main():
    if False:
        print('Hello World!')
    bar: Name = foo(1, 2, 3)
with logger.catch():
    main()