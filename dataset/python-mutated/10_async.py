"""This program is self-checking!"""
import asyncio
from contextlib import asynccontextmanager, AbstractAsyncContextManager
import functools

def _async_test(func):
    if False:
        i = 10
        return i + 15
    'Decorator to turn an async function into a test case.'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        coro = func(*args, **kwargs)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    return wrapper
state = []

@asynccontextmanager
async def woohoo():
    state.append(1)
    yield 42
    state.append(999)

@_async_test
async def test_enter():

    class DefaultEnter(AbstractAsyncContextManager):

        async def __aexit__(*args):
            return
    manager = DefaultEnter()
    got_manager = await manager.__aenter__()
    assert got_manager is manager
    async with manager as context:
        async with woohoo() as x:
            x = 1
            y = 2
        assert manager is context

class CoroutineTest:

    def test_with_8(self):
        if False:
            print('Hello World!')
        CNT = 0

        async def foo():
            nonlocal CNT
            async with CM():
                CNT += 1
                return
test_enter()