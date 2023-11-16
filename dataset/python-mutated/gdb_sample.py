def foo(a, b, c):
    if False:
        print('Hello World!')
    bar(a, b, c)

def bar(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    baz(a, b, c)

def baz(*args):
    if False:
        while True:
            i = 10
    id(42)
foo(1, 2, 3)