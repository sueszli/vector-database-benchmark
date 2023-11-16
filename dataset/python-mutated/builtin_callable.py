print(callable(None))
print(callable(1))
print(callable([]))
print(callable('dfsd'))
import sys
print(callable(sys))
print(callable(callable))
print(callable(lambda : None))

def f():
    if False:
        while True:
            i = 10
    pass
print(callable(f))

class A:
    pass
print(callable(A))
print(callable(A()))

class B:

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        pass
print(callable(B()))

class C:

    def f(self):
        if False:
            i = 10
            return i + 15
        return 'A.f'

class D:
    g = C()
print(callable(D().g))
print(D().g.f())