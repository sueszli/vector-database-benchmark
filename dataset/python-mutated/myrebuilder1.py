class A:

    def a(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'a'

class B(A):

    def b(self) -> str:
        if False:
            while True:
                i = 10
        return 'b'

class Inherit(A):

    def a(self) -> str:
        if False:
            while True:
                i = 10
        return 'c'