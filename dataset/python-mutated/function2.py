def f(a, **kwargs) -> A:
    if False:
        for i in range(10):
            print('nop')
    with cache_dir():
        if something:
            result = CliRunner().invoke(black.main, [str(src1), str(src2), '--diff', '--check'])
    limited.append(-limited.pop())
    return A(very_long_argument_name1=very_long_value_for_the_argument, very_long_argument_name2=-very.long.value.for_the_argument, **kwargs)

def g():
    if False:
        print('Hello World!')
    'Docstring.'

    def inner():
        if False:
            return 10
        pass
    print('Inner defs should breathe a little.')

def h():
    if False:
        for i in range(10):
            print('nop')

    def inner():
        if False:
            i = 10
            return i + 15
        pass
    print('Inner defs should breathe a little.')
if os.name == 'posix':
    import termios

    def i_should_be_followed_by_only_one_newline():
        if False:
            while True:
                i = 10
        pass
elif os.name == 'nt':
    try:
        import msvcrt

        def i_should_be_followed_by_only_one_newline():
            if False:
                print('Hello World!')
            pass
    except ImportError:

        def i_should_be_followed_by_only_one_newline():
            if False:
                while True:
                    i = 10
            pass
elif False:

    class IHopeYouAreHavingALovelyDay:

        def __call__(self):
            if False:
                return 10
            print('i_should_be_followed_by_only_one_newline')
else:

    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
with hmm_but_this_should_get_two_preceding_newlines():
    pass

def f(a, **kwargs) -> A:
    if False:
        i = 10
        return i + 15
    with cache_dir():
        if something:
            result = CliRunner().invoke(black.main, [str(src1), str(src2), '--diff', '--check'])
    limited.append(-limited.pop())
    return A(very_long_argument_name1=very_long_value_for_the_argument, very_long_argument_name2=-very.long.value.for_the_argument, **kwargs)

def g():
    if False:
        for i in range(10):
            print('nop')
    'Docstring.'

    def inner():
        if False:
            return 10
        pass
    print('Inner defs should breathe a little.')

def h():
    if False:
        for i in range(10):
            print('nop')

    def inner():
        if False:
            while True:
                i = 10
        pass
    print('Inner defs should breathe a little.')
if os.name == 'posix':
    import termios

    def i_should_be_followed_by_only_one_newline():
        if False:
            print('Hello World!')
        pass
elif os.name == 'nt':
    try:
        import msvcrt

        def i_should_be_followed_by_only_one_newline():
            if False:
                i = 10
                return i + 15
            pass
    except ImportError:

        def i_should_be_followed_by_only_one_newline():
            if False:
                i = 10
                return i + 15
            pass
elif False:

    class IHopeYouAreHavingALovelyDay:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            print('i_should_be_followed_by_only_one_newline')
else:

    def foo():
        if False:
            while True:
                i = 10
        pass
with hmm_but_this_should_get_two_preceding_newlines():
    pass