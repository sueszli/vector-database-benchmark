"""Test case for fixing F841 violations."""

def f():
    if False:
        i = 10
        return i + 15
    x = 1
    y = 2
    z = 3
    print(z)

def f():
    if False:
        for i in range(10):
            print('nop')
    x: int = 1
    y: int = 2
    z: int = 3
    print(z)

def f():
    if False:
        i = 10
        return i + 15
    with foo() as x1:
        pass
    with foo() as (x2, y2):
        pass
    with foo() as x3, foo() as y3, foo() as z3:
        pass

def f():
    if False:
        print('Hello World!')
    (x1, y1) = (1, 2)
    (x2, y2) = coords2 = (1, 2)
    coords3 = (x3, y3) = (1, 2)

def f():
    if False:
        for i in range(10):
            print('nop')
    try:
        1 / 0
    except ValueError as x1:
        pass
    try:
        1 / 0
    except (ValueError, ZeroDivisionError) as x2:
        pass

def f(a, b):
    if False:
        i = 10
        return i + 15
    x = a() if a is not None else b
    y = a() if a is not None else b

def f(a, b):
    if False:
        print('Hello World!')
    x = a if a is not None else b
    y = a if a is not None else b

def f():
    if False:
        return 10
    with Nested(m) as cm:
        pass

def f():
    if False:
        i = 10
        return i + 15
    with Nested(m) as cm:
        pass

def f():
    if False:
        i = 10
        return i + 15
    with Nested(m) as (x, y):
        pass

def f():
    if False:
        return 10
    with Nested(m) as cm:
        pass

def f():
    if False:
        i = 10
        return i + 15
    toplevel = tt = lexer.get_token()
    if not tt:
        break

def f():
    if False:
        while True:
            i = 10
    toplevel = tt = lexer.get_token()

def f():
    if False:
        return 10
    toplevel = (a, b) = lexer.get_token()

def f():
    if False:
        while True:
            i = 10
    (a, b) = toplevel = lexer.get_token()

def f():
    if False:
        return 10
    toplevel = tt = 1

def f(provided: int) -> int:
    if False:
        i = 10
        return i + 15
    match provided:
        case [_, *x]:
            pass

def f(provided: int) -> int:
    if False:
        while True:
            i = 10
    match provided:
        case x:
            pass

def f(provided: int) -> int:
    if False:
        return 10
    match provided:
        case Foo(bar) as x:
            pass

def f(provided: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    match provided:
        case {'foo': 0, **x}:
            pass

def f(provided: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    match provided:
        case {**x}:
            pass
global CONSTANT

def f() -> None:
    if False:
        print('Hello World!')
    global CONSTANT
    CONSTANT = 1
    CONSTANT = 2

def f() -> None:
    if False:
        i = 10
        return i + 15
    try:
        print('hello')
    except A as e:
        print('oh no!')

def f():
    if False:
        i = 10
        return i + 15
    x = 1
    y = 2

def f():
    if False:
        return 10
    x = 1
    y = 2

def f():
    if False:
        while True:
            i = 10
    x = foo()
    x = foo()
    x = y.z = foo()