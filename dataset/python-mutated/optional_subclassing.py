from __future__ import print_function

@cython.cclass
class A:

    @cython.cfunc
    def foo(self):
        if False:
            return 10
        print('A')

@cython.cclass
class B(A):

    @cython.cfunc
    def foo(self, x=None):
        if False:
            for i in range(10):
                print('nop')
        print('B', x)

@cython.cclass
class C(B):

    @cython.ccall
    def foo(self, x=True, k: cython.int=3):
        if False:
            return 10
        print('C', x, k)