"""Test case expected to be run with `suppress_dummy_args = True`."""

def foo(_) -> None:
    if False:
        i = 10
        return i + 15
    ...

def foo(*_) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def foo(**_) -> None:
    if False:
        while True:
            i = 10
    ...

def foo(a: int, _) -> None:
    if False:
        return 10
    ...

def foo() -> None:
    if False:
        return 10

    def bar(_) -> None:
        if False:
            return 10
        ...