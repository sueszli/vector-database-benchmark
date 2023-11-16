"""Test cases for test_pyclbr.py"""

def f():
    if False:
        return 10
    pass

class Other(object):

    @classmethod
    def foo(c):
        if False:
            while True:
                i = 10
        pass

    def om(self):
        if False:
            return 10
        pass

class B(object):

    def bm(self):
        if False:
            while True:
                i = 10
        pass

class C(B):
    foo = Other().foo
    om = Other.om
    d = 10

    def m(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def sm(self):
        if False:
            return 10
        pass

    @classmethod
    def cm(self):
        if False:
            return 10
        pass