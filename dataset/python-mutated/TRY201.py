"""
Violation:

Raising an exception using its assigned name is verbose and unrequired
"""
import logging
logger = logging.getLogger(__name__)

class MyException(Exception):
    pass

def bad():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException as e:
        logger.exception('process failed')
        raise e

def good():
    if False:
        print('Hello World!')
    try:
        process()
    except MyException:
        logger.exception('process failed')
        raise

def still_good():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException as e:
        print(e)
        raise

def still_good_too():
    if False:
        for i in range(10):
            print('nop')
    try:
        process()
    except MyException as e:
        print(e)
        raise e from None

def still_actually_good():
    if False:
        while True:
            i = 10
    try:
        process()
    except MyException as e:
        try:
            pass
        except TypeError:
            raise e

def bad_that_needs_recursion():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException as e:
        logger.exception('process failed')
        if True:
            raise e

def bad_that_needs_recursion_2():
    if False:
        i = 10
        return i + 15
    try:
        process()
    except MyException as e:
        logger.exception('process failed')
        if True:

            def foo():
                if False:
                    return 10
                raise e