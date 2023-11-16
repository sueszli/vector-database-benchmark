import inspect
from unittest import TestCase
import pytest
from hypothesis import example, given
from hypothesis.strategies import booleans, integers

def test_must_use_result_of_test():
    if False:
        i = 10
        return i + 15

    class DoubleRun:

        def execute_example(self, function):
            if False:
                return 10
            x = function()
            if inspect.isfunction(x):
                return x()

        @given(booleans())
        def boom(self, b):
            if False:
                for i in range(10):
                    print('nop')

            def f():
                if False:
                    while True:
                        i = 10
                raise ValueError
            return f
    with pytest.raises(ValueError):
        DoubleRun().boom()

class TestTryReallyHard(TestCase):

    @given(integers())
    def test_something(self, i):
        if False:
            print('Hello World!')
        pass

    def execute_example(self, f):
        if False:
            print('Hello World!')
        f()
        return f()

class Valueless:

    def execute_example(self, f):
        if False:
            for i in range(10):
                print('nop')
        try:
            return f()
        except ValueError:
            return None

    @given(integers())
    @example(1)
    def test_no_boom_on_example(self, x):
        if False:
            print('Hello World!')
        raise ValueError

    @given(integers())
    def test_no_boom(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError

    @given(integers())
    def test_boom(self, x):
        if False:
            for i in range(10):
                print('nop')
        raise AssertionError

def test_boom():
    if False:
        return 10
    with pytest.raises(AssertionError):
        Valueless().test_boom()

def test_no_boom():
    if False:
        for i in range(10):
            print('nop')
    Valueless().test_no_boom()

def test_no_boom_on_example():
    if False:
        while True:
            i = 10
    Valueless().test_no_boom_on_example()