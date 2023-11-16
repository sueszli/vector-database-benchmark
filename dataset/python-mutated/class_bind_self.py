class A:

    def __init__(self, arg):
        if False:
            print('Hello World!')
        self.val = arg

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'A.__str__ ' + str(self.val)

    def __call__(self, arg):
        if False:
            return 10
        return ('A.__call__', arg)

    def foo(self, arg):
        if False:
            while True:
                i = 10
        return ('A.foo', self.val, arg)

def make_closure(x_in):
    if False:
        i = 10
        return i + 15
    x = x_in

    def closure(y):
        if False:
            while True:
                i = 10
        return (x, y is c)
    return closure

class C:

    def f1(self, arg):
        if False:
            while True:
                i = 10
        return ('C.f1', self is c, arg)
    f2 = lambda self, arg: ('C.f2', self is c, arg)
    f3 = make_closure('f3')

    def f4(self, arg):
        if False:
            while True:
                i = 10
        yield (self is c, arg)
    f5 = int
    f6 = abs
    f7 = A
    f8 = A(8)
    f9 = A(9).foo
c = C()
print(c.f1(1))
print(c.f2(2))
print(c.f3())
print(next(c.f4(4)))
print(c.f5(5))
print(c.f6(-6))
print(c.f7(7))
print(c.f8(8))
print(c.f9(9))
print(C.f5(10))
print(C.f6(-11))
print(C.f7(12))
print(C.f8(13))
print(C.f9(14))