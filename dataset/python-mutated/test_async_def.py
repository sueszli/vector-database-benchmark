import asyncio
from unittest import TestCase
from hypothesis import assume, given, strategies as st

class TestAsyncioRun(TestCase):
    timeout = 5

    def execute_example(self, f):
        if False:
            i = 10
            return i + 15
        asyncio.run(f())

    @given(st.text())
    async def test_foo(self, x):
        assume(x)
        await asyncio.sleep(0.001)
        assert x