def foo(a, b=3):
    if False:
        i = 10
        return i + 15
    print(a, b)
foo(1, 333)
foo(1, b=333)
foo(a=2, b=333)

def foo2(a=1, b=2):
    if False:
        return 10
    print(a, b)
foo2(b='two')