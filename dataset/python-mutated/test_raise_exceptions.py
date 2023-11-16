"""Raising Exceptions.

@see: https://docs.python.org/3/tutorial/errors.html#raising-exceptions

The raise statement allows the programmer to force a specified exception to occur.
"""

def test_raise_exception():
    if False:
        i = 10
        return i + 15
    'Raising Exceptions.\n\n    The raise statement allows the programmer to force a specified exception to occur.\n    '
    exception_is_caught = False
    try:
        raise NameError('HiThere')
    except NameError:
        exception_is_caught = True
    assert exception_is_caught

def test_user_defined_exception():
    if False:
        return 10
    'User-defined Exceptions'

    class MyCustomError(Exception):
        """Example of MyCustomError exception."""

        def __init__(self, message):
            if False:
                return 10
            super().__init__(message)
            self.message = message
    custom_exception_is_caught = False
    try:
        raise MyCustomError('My custom message')
    except MyCustomError:
        custom_exception_is_caught = True
    assert custom_exception_is_caught