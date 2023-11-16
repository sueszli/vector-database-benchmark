class A:

    def foo(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 10

    @staticmethod
    def bar():
        if False:
            return 10
        return 42

class B(A):
    pass

class C(B):

    def baz(self):
        if False:
            for i in range(10):
                print('nop')
        return self.foo(10)
x = B().foo(10)
y = C.bar()