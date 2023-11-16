class A:

    def foo(self):
        if False:
            while True:
                i = 10
        print('A.foo')

class B:

    def foo(self):
        if False:
            return 10
        print('B.foo')

class C(A, B):

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        print('C.foo')
        super().foo()
C().foo()