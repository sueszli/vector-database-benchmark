"""
This is a docstring.

And this is a multi-line line: [http://example.com]
(https://example.com/blah/blah/blah.html).
"""
from dataclasses import dataclass
SOME_GLOBAL_VAR = "Ahhhh I'm a global var!!"
'\nThis is a global var.\n'

def func_with_no_args():
    if False:
        for i in range(10):
            print('nop')
    '\n    This function has no args.\n    '
    return None

def func_with_args(a: int, b: int, c: int=3) -> int:
    if False:
        return 10
    '\n    This function has some args.\n\n    # Parameters\n\n    a : `int`\n        A number.\n    b : `int`\n        Another number.\n    c : `int`, optional (default = `3`)\n        Yet another number.\n\n    Notes\n    -----\n\n    These are some notes.\n\n    # Returns\n\n    `int`\n        The result of `a + b * c`.\n    '
    return a + b * c

class SomeClass:
    """
    I'm a class!

    # Parameters

    x : `float`
        This attribute is called `x`.
    """
    some_class_level_variable = 1
    '\n    This is how you document a class-level variable.\n    '
    some_class_level_var_with_type: int = 1

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.x = 1.0

    def _private_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Private methods should not be included in documentation.\n        '
        pass

    def some_method(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        I'm a method!\n\n        But I don't do anything.\n\n        # Returns\n\n        `None`\n        "
        return None

    def method_with_alternative_return_section(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Another method.\n\n        # Returns\n\n        A completely arbitrary number.\n        '
        return 3

    def method_with_alternative_return_section3(self) -> int:
        if False:
            print('Hello World!')
        '\n        Another method.\n\n        # Returns\n\n        number : `int`\n            A completely arbitrary number.\n        '
        return 3

class AnotherClassWithReallyLongConstructor:

    def __init__(self, a_really_long_argument_name: int=0, another_long_name: float=2, these_variable_names_are_terrible: str='yea I know', **kwargs) -> None:
        if False:
            return 10
        self.a = a_really_long_argument_name
        self.b = another_long_name
        self.c = these_variable_names_are_terrible
        self.other = kwargs

@dataclass
class ClassWithDecorator:
    x: int

class _PrivateClass:

    def public_method_on_private_class(self):
        if False:
            return 10
        '\n        This should not be documented since the class is private.\n        '
        pass