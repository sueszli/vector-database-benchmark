"""Check that the constants are on the right side of the comparisons"""

class MyClass:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.attr = 1

    def dummy_return(self):
        if False:
            while True:
                i = 10
        return self.attr

def dummy_return():
    if False:
        print('Hello World!')
    return 2

def bad_comparisons():
    if False:
        for i in range(10):
            print('nop')
    'this is not ok'
    instance = MyClass()
    for i in range(10):
        if 5 <= i:
            pass
        if 1 == i:
            pass
        if 3 < dummy_return():
            pass
        if 4 != instance.dummy_return():
            pass
        if 1 == instance.attr:
            pass
        if 'aaa' == instance.attr:
            pass

def good_comparison():
    if False:
        return 10
    'this is ok'
    for i in range(10):
        if i == 5:
            pass

def double_comparison():
    if False:
        return 10
    'Check that we return early for non-binary comparison'
    for i in range(10):
        if i == 1 == 2:
            pass
        if 2 <= i <= 8:
            print('Between 2 and 8 inclusive')

def const_comparison():
    if False:
        return 10
    'Check that we return early for comparison of two constants'
    if 1 == 2:
        pass