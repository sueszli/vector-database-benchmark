"""
Violation:
Reraise without using 'from'
"""

class MyException(Exception):
    pass

def func():
    if False:
        for i in range(10):
            print('nop')
    try:
        a = 1
    except Exception:
        raise MyException()

def func():
    if False:
        return 10
    try:
        a = 1
    except Exception:
        if True:
            raise MyException()

def good():
    if False:
        i = 10
        return i + 15
    try:
        a = 1
    except MyException as e:
        raise e
    except Exception:
        raise

def good():
    if False:
        while True:
            i = 10
    try:
        from mod import f
    except ImportError:

        def f():
            if False:
                while True:
                    i = 10
            raise MyException()