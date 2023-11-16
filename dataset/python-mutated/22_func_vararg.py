def foo1(*args):
    if False:
        i = 10
        return i + 15
    func(args)

def foo2(**kwargs):
    if False:
        return 10
    func(kwargs)

def foo3(a, *args, **kw):
    if False:
        for i in range(10):
            print('nop')
    func(a, args, kw)