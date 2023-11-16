from __future__ import print_function
import sys
__class__ = 'Using module level __class__ variable, would be wrong for Python3'

class ClassWithUnderClassClosure:

    def g(self):
        if False:
            return 10

        def h():
            if False:
                i = 10
                return i + 15
            print('Variable __class__ in ClassWithUnderClassClosure is', __class__)
        h()
        try:
            print('ClassWithUnderClassClosure: Super in ClassWithUnderClassClosure is', super())
        except Exception as e:
            print('ClassWithUnderClassClosure: Occurred during super call', repr(e))
print('Class with a method that has a local function accessing __class__:')
ClassWithUnderClassClosure().g()

class ClassWithoutUnderClassClosure:

    def g(self):
        if False:
            for i in range(10):
                print('nop')
        __class__ = 'Providing __class__ ourselves, then it must be used'
        print(__class__)
        try:
            print('ClassWithoutUnderClassClosure: Super', super())
        except Exception as e:
            print('ClassWithoutUnderClassClosure: Occurred during super call', repr(e))
ClassWithoutUnderClassClosure().g()
__class__ = 'Global __class__'

def deco(C):
    if False:
        for i in range(10):
            print('nop')
    print('Decorating', repr(C))

    class D(C):
        pass
    return D

@deco
class X:
    __class__ = 'some string'

    def f1(self):
        if False:
            i = 10
            return i + 15
        print('f1', locals())
        try:
            print('f1', __class__)
        except Exception as e:
            print('Accessing __class__ in f1 gave', repr(e))

    def f2(self):
        if False:
            while True:
                i = 10
        print('f2', locals())

    def f4(self):
        if False:
            print('Hello World!')
        print('f4', self)
        self = X()
        print('f4', self)
        try:
            print('f4', super())
            print('f4', super().__self__)
        except TypeError:
            import sys
            assert sys.version_info < (3,)
    f5 = lambda x: __class__

    def f6(self_by_another_name):
        if False:
            for i in range(10):
                print('nop')
        try:
            print('f6', super())
        except TypeError:
            import sys
            assert sys.version_info < (3,)

    def f7(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            yield super()
        except TypeError:
            import sys
            assert sys.version_info < (3,)
    print('Early pre-class calls begin')
    print('Set in class __class__', __class__)
    f2(2)
    print('Early pre-class calls end')
    del __class__
x = X()
x.f1()
x.f2()
x.f4()
print('f5', x.f5())
x.f6()
print('f7', list(x.f7()))

def makeSuperCall(arg1, arg2):
    if False:
        for i in range(10):
            print('nop')
    print('Calling super with args', arg1, arg2, end=': ')
    try:
        super(arg1, arg2)
    except Exception as e:
        print('Exception', e)
    else:
        print('Ok.')
if sys.version_info >= (3, 6):
    makeSuperCall(None, None)
    makeSuperCall(1, None)
makeSuperCall(type, None)
makeSuperCall(type, 1)