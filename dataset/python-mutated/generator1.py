def f(x):
    if False:
        print('Hello World!')
    print('a')
    y = x
    print('b')
    while y > 0:
        print('c')
        y -= 1
        print('d')
        yield y
        print('e')
    print('f')
    return None
for val in f(3):
    print(val)
print(repr(f(0))[0:17])