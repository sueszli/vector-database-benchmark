"""
Input for test_profile.py and test_cprofile.py.

IMPORTANT: This stuff is touchy. If you modify anything above the
test class you'll have to regenerate the stats by running the two
test files.

*ALL* NUMBERS in the expected output are relevant.  If you change
the formatting of pstats, please don't just regenerate the expected
output without checking very carefully that not a single number has
changed.
"""
import sys
TICKS = 42000

def timer():
    if False:
        return 10
    return TICKS

def testfunc():
    if False:
        i = 10
        return i + 15
    global TICKS
    TICKS += 99
    helper()
    helper()
    TICKS += 171
    factorial(14)

def factorial(n):
    if False:
        i = 10
        return i + 15
    global TICKS
    if n > 0:
        TICKS += n
        return mul(n, factorial(n - 1))
    else:
        TICKS += 11
        return 1

def mul(a, b):
    if False:
        i = 10
        return i + 15
    global TICKS
    TICKS += 1
    return a * b

def helper():
    if False:
        print('Hello World!')
    global TICKS
    TICKS += 1
    helper1()
    TICKS += 2
    helper1()
    TICKS += 6
    helper2()
    TICKS += 3
    helper2()
    TICKS += 2
    helper2()
    TICKS += 5
    helper2_indirect()
    TICKS += 1

def helper1():
    if False:
        while True:
            i = 10
    global TICKS
    TICKS += 10
    hasattr(C(), 'foo')
    TICKS += 19
    lst = []
    lst.append(42)
    sys.exc_info()

def helper2_indirect():
    if False:
        while True:
            i = 10
    helper2()
    factorial(3)

def helper2():
    if False:
        i = 10
        return i + 15
    global TICKS
    TICKS += 11
    hasattr(C(), 'bar')
    TICKS += 13
    subhelper()
    TICKS += 15

def subhelper():
    if False:
        print('Hello World!')
    global TICKS
    TICKS += 2
    for i in range(2):
        try:
            C().foo
        except AttributeError:
            TICKS += 3

class C:

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        global TICKS
        TICKS += 1
        raise AttributeError