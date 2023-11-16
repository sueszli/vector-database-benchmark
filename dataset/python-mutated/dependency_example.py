"""
A file for testing the gathering of sources and dependency by test_dependencies
"""
import mock
import pytest
from tests.foo import bar, mock_extension

def some_func():
    if False:
        for i in range(10):
            print('nop')
    pass
ignore_this = 17