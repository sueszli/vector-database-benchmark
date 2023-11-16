import asyncio
import warnings
from unittest import TestCase
import pytest
from hypothesis import assume, given, strategies as st
from hypothesis.internal.compat import PYPY

def coro_decorator(f):
    if False:
        i = 10
        return i + 15
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DeprecationWarning)
        return asyncio.coroutine(f)

class TestAsyncio(TestCase):
    timeout = 5

    def setUp(self):
        if False:
            while True:
                i = 10
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        if False:
            return 10
        self.loop.close()

    def execute_example(self, f):
        if False:
            return 10
        error = None

        def g():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal error
            try:
                x = f()
                if x is not None:
                    yield from x
            except BaseException as e:
                error = e
        coro = coro_decorator(g)
        future = asyncio.wait_for(coro(), timeout=self.timeout)
        self.loop.run_until_complete(future)
        if error is not None:
            raise error

    @pytest.mark.skipif(PYPY, reason='Error in asyncio.new_event_loop()')
    @given(x=st.text())
    @coro_decorator
    def test_foo(self, x):
        if False:
            print('Hello World!')
        assume(x)
        yield from asyncio.sleep(0.001)
        assert x

class TestAsyncioRun(TestCase):

    def execute_example(self, f):
        if False:
            for i in range(10):
                print('nop')
        asyncio.run(f())

    @given(x=st.text())
    @coro_decorator
    def test_foo(self, x):
        if False:
            for i in range(10):
                print('nop')
        assume(x)
        yield from asyncio.sleep(0.001)
        assert x