import dill
dill.settings['recurse'] = True

def f(func):
    if False:
        return 10

    def w(*args):
        if False:
            print('Hello World!')
        return f(*args)
    return w

@f
def f2():
    if False:
        while True:
            i = 10
    pass

def test_decorated():
    if False:
        while True:
            i = 10
    assert dill.pickles(f2)
import doctest
import logging
logging.basicConfig(level=logging.DEBUG)

class SomeUnreferencedUnpicklableClass(object):

    def __reduce__(self):
        if False:
            print('Hello World!')
        raise Exception
unpicklable = SomeUnreferencedUnpicklableClass()

def test_normal():
    if False:
        for i in range(10):
            print('nop')
    serialized = dill.dumps(lambda x: x)

def tests():
    if False:
        print('Hello World!')
    '\n    >>> serialized = dill.dumps(lambda x: x)\n    '
    return

def test_doctest():
    if False:
        print('Hello World!')
    doctest.testmod()
if __name__ == '__main__':
    test_decorated()
    test_normal()
    test_doctest()