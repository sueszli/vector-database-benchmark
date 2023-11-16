class A:

    def foo(self):
        if False:
            while True:
                i = 10
        return 'A'

class B(A):

    def foo(self):
        if False:
            while True:
                i = 10
        return 'B'