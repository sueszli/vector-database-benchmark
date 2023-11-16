""" Exercise assertions in their flavors."""
from __future__ import print_function

def testAssert1():
    if False:
        return 10
    assert False
    return 1

def testAssert2():
    if False:
        print('Hello World!')
    assert True
    return 1

def testAssert3():
    if False:
        while True:
            i = 10
    assert False, 'argument'
    return 1
try:
    print('Function that will assert.')
    testAssert1()
    print('No exception.')
except Exception as e:
    print('Raised', type(e), e)
try:
    print('Function that will not assert.')
    testAssert2()
    print('No exception.')
except Exception as e:
    print('Raised', type(e), e)
try:
    print('Function that will assert with argument.')
    testAssert3()
    print('No exception.')
except Exception as e:
    print('Raised', type(e), e)
try:
    print('Assertion with tuple argument.', end='')
    assert False, (3,)
except AssertionError as e:
    print(str(e))
try:
    print('Assertion with plain argument.', end='')
    assert False, 3
except AssertionError as e:
    print(str(e))