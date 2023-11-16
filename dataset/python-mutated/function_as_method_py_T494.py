class A:
    """
    >>> A.foo = foo
    >>> A().foo()
    True
    """
    pass

def foo(self):
    if False:
        print('Hello World!')
    return self is not None

def f_plus(a):
    if False:
        for i in range(10):
            print('nop')
    return a + 1

class B:
    """
    >>> B.plus1(1)
    2
    """
    plus1 = f_plus

class C(object):
    """
    >>> C.plus1(1)
    2
    """
    plus1 = f_plus