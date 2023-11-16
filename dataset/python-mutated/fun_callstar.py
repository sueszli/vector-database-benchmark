def foo(a, b, c):
    if False:
        print('Hello World!')
    print(a, b, c)
foo(*(), 1, 2, 3)
foo(*(1,), 2, 3)
foo(*(1, 2), 3)
foo(*(1, 2, 3))
foo(1, *(2, 3))
foo(1, 2, *(3,))
foo(1, 2, 3, *())
foo(*(1,), 2, *(3,))
foo(*(1, 2), *(3,))
foo(*(1,), *(2, 3))
foo(1, 2, *[100])
foo(*range(3))
foo(1, *range(2, 4))

def foo(*rest):
    if False:
        print('Hello World!')
    print(rest)
foo(*range(10))

class A:

    def foo(self, a, b, c):
        if False:
            while True:
                i = 10
        print(a, b, c)
a = A()
a.foo(*(), 1, 2, 3)
a.foo(*(1,), 2, 3)
a.foo(*(1, 2), 3)
a.foo(*(1, 2, 3))
a.foo(1, *(2, 3))
a.foo(1, 2, *(3,))
a.foo(1, 2, 3, *())
a.foo(*(1,), 2, *(3,))
a.foo(*(1, 2), *(3,))
a.foo(*(1,), *(2, 3))
a.foo(1, 2, *[100])
a.foo(*range(3))