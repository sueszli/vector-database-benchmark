def f(name, args):
    if False:
        for i in range(10):
            print('nop')
    return f'foo.{name!r}{name!s}{name!a}'