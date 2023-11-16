"""
Exception classes are used to indicate that something has gone wrong with
the program at runtime. Functions use the `raise` keyword, if an error is
anticipated, and specify the exception class they intend to throw. This
module defines a handful of custom exception classes and shows how they
can be used in the context of a function.
"""

class CustomError(Exception):
    """Custom class of errors.

    This is a custom exception for any issues that arise in this module.
    One of the reasons why developers design a class like this is for
    consumption by downstream services and command-line tools.

    If we designed a standalone application with no downstream consumers, then
    it makes little sense to define a custom hierarchy of exceptions. In that
    case, we should use the existing hierarchy of builtin exception
    classes which are listed in the Python docs:

    https://docs.python.org/3/library/exceptions.html
    """

class DivisionError(CustomError):
    """Any division error that results from invalid input.

    This exception can be subclassed with the following exceptions if they
    happen enough across the codebase:

    - ZeroDivisorError
    - NegativeDividendError
    - NegativeDivisorError

    That being said, there's a point of diminishing returns when we design
    too many exceptions. It is better to design few exceptions that many
    developers handle than design many exceptions that few developers handle.
    """

def divide_positive_numbers(dividend, divisor):
    if False:
        return 10
    'Divide a positive number by another positive number.\n\n    Writing a program in this style is considered defensive programming.\n    For more on this programming style, check the Wikipedia link below:\n\n    https://en.wikipedia.org/wiki/Defensive_programming\n    '
    if dividend <= 0:
        raise DivisionError(f'Non-positive dividend: {dividend}')
    elif divisor <= 0:
        raise DivisionError(f'Non-positive divisor: {divisor}')
    return dividend // divisor

def main():
    if False:
        print('Hello World!')
    assert issubclass(DivisionError, CustomError)
    for (dividend, divisor) in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
        division_failed = False
        try:
            divide_positive_numbers(dividend, divisor)
        except DivisionError as e:
            division_failed = True
            assert str(e).startswith('Non-positive')
        assert division_failed is True
    result = divide_positive_numbers(1, 1)
    assert result == 1
if __name__ == '__main__':
    main()