"""
Correct syntax for variable annotation that should fail at runtime
in a certain manner. More examples are in test_grammar and test_parser.
"""

def f_bad_ann():
    if False:
        while True:
            i = 10
    __annotations__[1] = 2

class C_OK:

    def __init__(self, x: int) -> None:
        if False:
            while True:
                i = 10
        self.x: no_such_name = x

class D_bad_ann:

    def __init__(self, x: int) -> None:
        if False:
            i = 10
            return i + 15
        sfel.y: int = 0

def g_bad_ann():
    if False:
        print('Hello World!')
    no_such_name.attr: int = 0