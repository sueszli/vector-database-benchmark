"""
Violation:

Returning a final value inside a try block may indicate you could use an else block
instead to outline the success scenario
"""
import logging
logger = logging.getLogger(__name__)

class MyException(Exception):
    pass

def bad():
    if False:
        print('Hello World!')
    try:
        a = 1
        b = process()
        return b
    except MyException:
        logger.exception('process failed')

def good():
    if False:
        i = 10
        return i + 15
    try:
        a = 1
        b = process()
    except MyException:
        logger.exception('process failed')
    else:
        return b

def noreturn():
    if False:
        print('Hello World!')
    try:
        a = 1
        b = process()
    except MyException:
        logger.exception('process failed')

def good_return_with_side_effects():
    if False:
        i = 10
        return i + 15
    try:
        pass
        return process()
    except MyException:
        logger.exception('process failed')

def good_noexcept():
    if False:
        for i in range(10):
            print('nop')
    try:
        pass
        return process()
    finally:
        logger.exception('process failed')