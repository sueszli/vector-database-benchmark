def foo(s):
    if False:
        for i in range(10):
            print('nop')
    n = int(s)
    if n == 0:
        raise ValueError('invalid value: %s' % s)
    return 10 / n

def bar():
    if False:
        for i in range(10):
            print('nop')
    try:
        foo('0')
    except ValueError as e:
        print('ValueError!')
        raise
bar()