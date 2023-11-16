import asyncio
from .common import StaticTestBase

class AsyncTests(StaticTestBase):

    def test_async_for(self) -> None:
        if False:
            return 10
        codestr = '\n        from typing import Awaitable, List\n        async def foo(awaitables) -> int:\n             sum = 0\n             async for x in awaitables:\n                 sum += x\n             return sum\n        '
        self.compile(codestr)

    def test_async_for_name_error(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        from typing import Awaitable, List\n        async def foo(awaitables) -> int:\n             sum = 0\n             async for x in awaitables:\n                 sum += y\n             return sum\n        '
        self.type_error(codestr, 'Name `y` is not defined.')

    def test_async_for_primitive_error(self) -> None:
        if False:
            return 10
        codestr = '\n        from __static__ import int64\n        from typing import Awaitable, List\n        async def foo() -> int:\n             awaitables: int64 = 1\n             async for x in awaitables:\n                 sum += x\n             return sum\n\n        async def asyncify(x):\n             return x\n        '
        self.type_error(codestr, 'cannot await a primitive value')

    def test_async_with(self) -> None:
        if False:
            return 10
        codestr = '\n        from typing import Awaitable, List\n        async def foo(acm) -> None:\n             async with acm() as c:\n                 c.m()\n        '
        self.compile(codestr)

    def test_async_with_name_error(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        from typing import Awaitable, List\n        async def foo(acm) -> int:\n             async with acm() as c:\n                 d.m()\n        '
        self.type_error(codestr, 'Name `d` is not defined.')

    def test_async_with_may_not_terminate(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        from typing import Awaitable, List\n        async def foo(acm) -> int:\n             async with acm() as c:\n                 return 42\n        '
        self.type_error(codestr, "Function has declared return type 'int' but can implicitly return None.")