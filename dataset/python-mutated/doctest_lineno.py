def func_with_docstring():
    if False:
        i = 10
        return i + 15
    'Some unrelated info.'

def func_without_docstring():
    if False:
        while True:
            i = 10
    pass

def func_with_doctest():
    if False:
        for i in range(10):
            print('nop')
    "\n    This function really contains a test case.\n\n    >>> func_with_doctest.__name__\n    'func_with_doctest'\n    "
    return 3

class ClassWithDocstring:
    """Some unrelated class information."""

class ClassWithoutDocstring:
    pass

class ClassWithDoctest:
    """This class really has a test case in it.

    >>> ClassWithDoctest.__name__
    'ClassWithDoctest'
    """

class MethodWrapper:

    def method_with_docstring(self):
        if False:
            return 10
        'Method with a docstring.'

    def method_without_docstring(self):
        if False:
            i = 10
            return i + 15
        pass

    def method_with_doctest(self):
        if False:
            while True:
                i = 10
        "\n        This has a doctest!\n        >>> MethodWrapper.method_with_doctest.__name__\n        'method_with_doctest'\n        "