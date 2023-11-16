try:
    1 / 0
except ValueError as e:
    pass
try:
    1 / 0
except ValueError as e:
    print(e)

def f():
    if False:
        return 10
    x = 1
    y = 2
    z = x + y

def f():
    if False:
        i = 10
        return i + 15
    foo = (1, 2)
    (a, b) = (1, 2)
    bar = (1, 2)
    (c, d) = bar
    (x, y) = baz = bar

def f():
    if False:
        print('Hello World!')
    locals()
    x = 1

def f():
    if False:
        while True:
            i = 10
    _ = 1
    __ = 1
    _discarded = 1
a = 1

def f():
    if False:
        return 10
    global a
    b = 1

    def c():
        if False:
            for i in range(10):
                print('nop')
        b = 1

    def d():
        if False:
            return 10
        nonlocal b

def f():
    if False:
        print('Hello World!')
    annotations = []
    assert len([annotations for annotations in annotations])

def f():
    if False:
        return 10

    def connect():
        if False:
            print('Hello World!')
        return (None, None)
    with connect() as (connection, cursor):
        cursor.execute('SELECT * FROM users')

def f():
    if False:
        print('Hello World!')

    def connect():
        if False:
            return 10
        return (None, None)
    with connect() as (connection, cursor):
        cursor.execute('SELECT * FROM users')

def f():
    if False:
        print('Hello World!')
    with open('file') as my_file, open('') as (this, that):
        print('hello')

def f():
    if False:
        return 10
    with open('file') as my_file, open('') as (this, that):
        print('hello')

def f():
    if False:
        return 10
    (exponential, base_multiplier) = (1, 2)
    hash_map = {(exponential := (exponential * base_multiplier % 3)): i + 1 for i in range(2)}
    return hash_map

def f(x: int):
    if False:
        print('Hello World!')
    msg1 = 'Hello, world!'
    msg2 = 'Hello, world!'
    msg3 = 'Hello, world!'
    match x:
        case 1:
            print(msg1)
        case 2:
            print(msg2)

def f(x: int):
    if False:
        for i in range(10):
            print('nop')
    import enum
    Foo = enum.Enum('Foo', 'A B')
    Bar = enum.Enum('Bar', 'A B')
    Baz = enum.Enum('Baz', 'A B')
    match x:
        case Foo.A:
            print('A')
        case [Bar.A, *_]:
            print('A')
        case y:
            pass

def f():
    if False:
        return 10
    if any(((key := (value := x)) for x in ['ok'])):
        print(key)

def f() -> None:
    if False:
        while True:
            i = 10
    is_connected = False

    class Foo:

        @property
        def is_connected(self):
            if False:
                while True:
                    i = 10
            nonlocal is_connected
            return is_connected

        def do_thing(self):
            if False:
                while True:
                    i = 10
            nonlocal is_connected
            print(is_connected)
    obj = Foo()
    obj.do_thing()

def f():
    if False:
        return 10
    try:
        pass
    except Exception as _:
        pass