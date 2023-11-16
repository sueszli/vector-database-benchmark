class A:
    """A(foo, bar)"""

class B:
    """B(foo, bar)"""

    def __init__(self):
        if False:
            return 10
        'B(foo, bar, baz)'

class C:
    """C(foo, bar)"""

    def __new__(cls):
        if False:
            for i in range(10):
                print('nop')
        'C(foo, bar, baz)'

class D:

    def __init__(self):
        if False:
            while True:
                i = 10
        'D(foo, bar, baz)'

class E:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'E(foo: int, bar: int, baz: int) -> None \\\n        E(foo: str, bar: str, baz: str) -> None \\\n        E(foo: float, bar: float, baz: float)'

class F:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'F(foo: int, bar: int, baz: int) -> None\n        F(foo: str, bar: str, baz: str) -> None\n        F(foo: float, bar: float, baz: float)'