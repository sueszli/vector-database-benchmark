"""This file provides helpers to detect particular running conditions and skip the test when appropriate."""

def skip():
    if False:
        i = 10
        return i + 15
    print('SKIP')
    raise SystemExit

def always():
    if False:
        for i in range(10):
            print('nop')
    skip()

def no_reversed():
    if False:
        for i in range(10):
            print('nop')
    import builtins
    if 'reversed' not in dir(builtins):
        skip()

def no_bigint():
    if False:
        print('Hello World!')
    try:
        x = 40
        x = 1 << x
    except OverflowError:
        skip()

def board_in(*board):
    if False:
        for i in range(10):
            print('nop')
    try:
        import test_env
    except ImportError:

        class Env:

            def __init__(self, board):
                if False:
                    print('Hello World!')
                self.board = board
        test_env = Env('unknown')
    if test_env.board in board:
        skip()

def board_not_in(*board):
    if False:
        i = 10
        return i + 15
    try:
        import test_env
    except ImportError:

        class Env:

            def __init__(self, board):
                if False:
                    print('Hello World!')
                self.board = board
        test_env = Env('unknown')
    if test_env.board not in board:
        skip()

def no_cpython_compat():
    if False:
        print('Hello World!')
    try:
        from collections import namedtuple
    except ImportError:
        skip()
    try:
        T3 = namedtuple('TupComma', 'foo bar')
    except TypeError:
        skip()

def no_slice_assign():
    if False:
        while True:
            i = 10
    try:
        memoryview
    except:
        skip()
    b1 = bytearray(b'1234')
    b2 = bytearray(b'5678')
    m1 = memoryview(b1)
    m2 = memoryview(b2)
    try:
        m2[1:3] = m1[0:2]
    except TypeError:
        skip()

def no_reverse_ops():
    if False:
        print('Hello World!')

    class Foo:

        def __radd__(self, other):
            if False:
                i = 10
                return i + 15
            pass
    try:
        5 + Foo()
    except TypeError:
        skip()