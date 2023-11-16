def check_syntax_error(statement):
    if False:
        print('Hello World!')
    try:
        compile(statement, '<test string>', 'exec')
    except SyntaxError:
        return
    assert False

def test_yield():
    if False:
        i = 10
        return i + 15

    def g():
        if False:
            print('Hello World!')
        f((yield 1), 1)

    def g():
        if False:
            print('Hello World!')
        f((yield from ()))

    def g():
        if False:
            i = 10
            return i + 15
        f((yield from ()), 1)

    def g():
        if False:
            for i in range(10):
                print('nop')
        f((yield 1))

    def g():
        if False:
            print('Hello World!')
        yield 1

    def g():
        if False:
            return 10
        yield from ()

    def g():
        if False:
            return 10
        x = (yield 1)

    def g():
        if False:
            print('Hello World!')
        x = (yield from ())

    def g():
        if False:
            for i in range(10):
                print('nop')
        yield (1, 1)

    def g():
        if False:
            print('Hello World!')
        x = (yield (1, 1))
    check_syntax_error('def g(): yield from (), 1')
    check_syntax_error('def g(): x = yield from (), 1')

    def g():
        if False:
            print('Hello World!')
        (1, (yield 1))

    def g():
        if False:
            return 10
        (1, (yield from ()))
    check_syntax_error('def g(): 1, yield 1')
    check_syntax_error('def g(): 1, yield from ()')

    def g():
        if False:
            print('Hello World!')
        f((yield 1))

    def g():
        if False:
            print('Hello World!')
        f((yield 1), 1)

    def g():
        if False:
            print('Hello World!')
        f((yield from ()))

    def g():
        if False:
            for i in range(10):
                print('nop')
        f((yield from ()), 1)
    check_syntax_error('def g(): f(yield 1)')
    check_syntax_error('def g(): f(yield 1, 1)')
    check_syntax_error('def g(): f(yield from ())')
    check_syntax_error('def g(): f(yield from (), 1)')
    check_syntax_error('yield')
    check_syntax_error('yield from')
    check_syntax_error('class foo:yield 1')
    check_syntax_error('class foo:yield from ()')
    check_syntax_error('def g(a:(yield)): pass')
test_yield()

def gen_func():
    if False:
        for i in range(10):
            print('nop')
    yield 1
    return (yield 2)
gen = gen_func()