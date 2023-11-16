"""Tests for compatibility of @inject-patched functions with asyncio and inspect module checks."""
import asyncio
import inspect
from dependency_injector.wiring import inject

def test_asyncio_iscoroutinefunction():
    if False:
        print('Hello World!')

    @inject
    async def foo():
        ...
    assert asyncio.iscoroutinefunction(foo)

def test_inspect_iscoroutinefunction():
    if False:
        while True:
            i = 10

    @inject
    async def foo():
        ...
    assert inspect.iscoroutinefunction(foo)