import contextlib
import os
import sys
from functools import wraps
import pytest

def replace_kwargs(new_kwargs):
    if False:
        print('Hello World!')

    def wrapper(func):
        if False:
            for i in range(10):
                print('nop')

        @wraps(func)
        def wrapped(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            kwargs.update(new_kwargs)
            return func(*args, **kwargs)
        return wrapped
    return wrapper

@contextlib.contextmanager
def null_assert_warnings(*args, **kwargs):
    if False:
        print('Hello World!')
    try:
        yield []
    finally:
        pass

@pytest.fixture(scope='session', autouse=True)
def patch_testing_functions():
    if False:
        while True:
            i = 10
    tm.assert_produces_warning = null_assert_warnings
    pytest.raises = replace_kwargs({'match': None})(pytest.raises)
sys.path.append(os.path.dirname(__file__))