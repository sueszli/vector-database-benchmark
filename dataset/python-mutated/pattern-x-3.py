def foo(x):
    if False:
        for i in range(10):
            print('nop')
    a = x + b
    c = a + x
    return c + x
bar = x
baz = 2 * bar