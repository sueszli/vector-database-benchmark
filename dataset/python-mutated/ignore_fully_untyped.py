"""Test case expected to be run with `ignore_fully_untyped = True`."""

def ok_fully_untyped_1(a, b):
    if False:
        return 10
    pass

def ok_fully_untyped_2():
    if False:
        i = 10
        return i + 15
    pass

def ok_fully_typed_1(a: int, b: int) -> int:
    if False:
        print('Hello World!')
    pass

def ok_fully_typed_2() -> int:
    if False:
        return 10
    pass

def ok_fully_typed_3(a: int, *args: str, **kwargs: str) -> int:
    if False:
        while True:
            i = 10
    pass

def error_partially_typed_1(a: int, b):
    if False:
        return 10
    pass

def error_partially_typed_2(a: int, b) -> int:
    if False:
        i = 10
        return i + 15
    pass

def error_partially_typed_3(a: int, b: int):
    if False:
        while True:
            i = 10
    pass

class X:

    def ok_untyped_method_with_arg(self, a):
        if False:
            i = 10
            return i + 15
        pass

    def ok_untyped_method(self):
        if False:
            print('Hello World!')
        pass

    def error_typed_self(self: X):
        if False:
            for i in range(10):
                print('nop')
        pass