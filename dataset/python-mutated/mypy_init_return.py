"""Test case expected to be run with `mypy_init_return = True`."""

class Foo:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        ...

class Foo:

    def __init__(self, foo):
        if False:
            return 10
        ...

class Foo:

    def __init__(self, foo) -> None:
        if False:
            print('Hello World!')
        ...

class Foo:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        ...

class Foo:

    def __init__(self, foo: int):
        if False:
            while True:
                i = 10
        ...

class Foo:

    def __init__(self, foo: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

def __init__(self, foo: int):
    if False:
        print('Hello World!')
    ...

class Foo:

    def __init__(self, *arg):
        if False:
            for i in range(10):
                print('nop')
        ...