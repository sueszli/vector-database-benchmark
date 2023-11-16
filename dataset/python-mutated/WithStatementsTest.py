from __future__ import print_function
import sys
x = 0

class MyContextManager(object):

    def __getattribute__(self, attribute_name):
        if False:
            while True:
                i = 10
        print('Asking context manager attribute', attribute_name)
        return object.__getattribute__(self, attribute_name)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        global x
        x += 1
        print('Entered context manager with counter value', x)
        return x

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        print('Context manager exit sees', exc_type, exc_value, traceback)
        print('Published to context manager exit is', sys.exc_info())
        return False
print('Use context manager and raise no exception in the body:')
with MyContextManager() as x:
    print('x has become', x)
print('Use context manager and raise an exception in the body:')
try:
    with MyContextManager() as x:
        print('x has become', x)
        raise Exception('Lalala')
        print(x)
except Exception as e:
    print('Caught raised exception', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
l = list(range(3))
print('Use context manager and assign to subscription target:')
with MyContextManager() as l[0]:
    print('Complex assignment target works', l[0])
try:
    with MyContextManager():
        sys.exit(9)
except BaseException as e:
    print('Caught base exception', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
print('Use context manager and fail to assign to attribute:')
try:
    with MyContextManager() as l.wontwork:
        sys.exit(9)
except BaseException as e:
    print('Caught base exception', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
print('Use context manager to do nothing inside:')
with MyContextManager() as x:
    pass
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)

def returnFromContextBlock():
    if False:
        return 10
    with MyContextManager() as x:
        return 7
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
print('Use context manager to return value:')
r = returnFromContextBlock()
print('Return value', r)

class NonContextManager1:

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

class NonContextManager2:

    def __exit__(self):
        if False:
            for i in range(10):
                print('nop')
        return self
print('Use incomplete context managers:')
try:
    with NonContextManager1() as x:
        print(x)
except Exception as e:
    print('Caught for context manager without __exit__', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
try:
    with NonContextManager2() as x:
        print(x)
except Exception as e:
    print('Caught for context manager without __enter__', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)

class NotAtAllContextManager:
    pass
try:
    with NotAtAllContextManager() as x:
        print(x)
except Exception as e:
    print('Caught for context manager without any special methods', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)

class MeanContextManager:

    def __enter__(self):
        if False:
            print('Hello World!')
        raise ValueError("Nah, I won't play")

    def __exit__(self):
        if False:
            return 10
        print('Called exit, yes')
print('Use mean context manager:')
try:
    with MeanContextManager() as x:
        print(x)
except Exception as e:
    print('Caught from mean manager', repr(e))
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)

class CatchingContextManager(object):

    def __enter__(self):
        if False:
            print('Hello World!')
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        return True
print('Suppressing exception from context manager body:')
with CatchingContextManager():
    raise ZeroDivisionError
if sys.version_info >= (3,):
    assert sys.exc_info() == (None, None, None)
print('OK')