class A:

    def a(self) -> str:
        if False:
            print('Hello World!')
        return 'b'

class B(A):

    def b(self) -> str:
        if False:
            return 10
        return 'c'

class Inherit(A):

    def a(self) -> str:
        if False:
            while True:
                i = 10
        return 'd'