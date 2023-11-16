""" Testing utilities, such as TestError, assert_value, assert_raises. """
from contextlib import contextmanager
from typing import NoReturn

class TestError(Exception):
    """
    Raised by assert_value and assert_raises, but may be manually raised
    to indicate a test error.
    """

def assert_value(value, expected=None, validator=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks 'value' for equality with 'expected', or, if validator is given,\n    checks bool(validator(value)) == True. Raises TestError on failure.\n\n    Example usage:\n\n    assert_value(fibonacci(0), 1)\n    "
    if expected is not None and validator is not None:
        raise ValueError("can't have both 'expected' and 'validator'")
    if validator is None:
        success = value == expected
    else:
        success = validator(value)
    if success:
        return
    raise TestError('unexpected result: ' + repr(value))

def result(value) -> NoReturn:
    if False:
        print('Hello World!')
    '\n    Shall be called when a result is unexpectedly returned in an assert_raises\n    block.\n    '
    raise TestError('expected exception, but got result: ' + repr(value))

@contextmanager
def assert_raises(expectedexception):
    if False:
        i = 10
        return i + 15
    '\n    Context guard that asserts that a certain exception is raised inside.\n\n    On successful execution (if the error failed to show up),\n    result() shall be called.\n\n    Example usage:\n\n    # we expect fibonacci to raise ValueError for negative values.\n    with assert_raises(ValueError):\n        result(fibonacci(-3))\n    '
    try:
        yield
    except expectedexception:
        return
    except TestError:
        raise
    except BaseException as exc:
        raise TestError('unexpected exception') from exc
    else:
        raise TestError('got neither an exception, nor a result')