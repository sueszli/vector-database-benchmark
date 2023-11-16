"""Tests for coroutines."""
from pytype.tests import test_base
from pytype.tests import test_utils

class GeneratorFeatureTest(test_base.BaseTest):
    """Tests for coroutines."""

    def test_ret_type_match(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Any, Awaitable, Coroutine, List\n\n      c: Coroutine[List[str], str, int] = None\n      async def data() -> str:\n        return 'data'\n\n      def f1() -> Awaitable[str]:\n        return data()\n\n      def f2() -> Coroutine[Any, Any, str]:\n        return data()\n\n      def f3() -> Coroutine[List[str], str, int]:\n        return c\n    ")

    def test_coroutine_typevar_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import List, Coroutine, Any\n\n      async def f() -> int:\n        return 1\n\n      c: Coroutine[Any, Any, int] = None\n      c = f()\n      x = c.send("str")\n      async def bar():\n        x = await c\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Coroutine\n\n      c: Coroutine[Any, Any, int]\n      x: Any\n\n      def bar() -> Coroutine[Any, Any, int]: ...\n      def f() -> Coroutine[Any, Any, int]: ...\n    ')

    def test_native_coroutine_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      async def callee():\n        if __random__:\n          return 1\n        else:\n          return "str"\n\n      async def caller():\n        x = await callee()\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Union, Coroutine\n\n      def callee() -> Coroutine[Any, Any, Union[int, str]]: ...\n      def caller() -> Coroutine[Any, Any, Union[int, str]]: ...\n    ')

    def test_native_coroutine_error(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      async def f1() -> str:\n        return 1  # bad-return-type[e1]\n\n      async def f2() -> int:\n        return 1\n\n      async def f3():\n        return 1\n\n      def f4(x: str):\n        pass\n\n      async def caller():\n        f4(await f1())\n        f4(await f2())  # wrong-arg-types[e2]\n        f4(await f3())  # wrong-arg-types[e3]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'str.*int', 'e3': 'str.*int'})

    def test_generator_based_coroutine_pyi(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import asyncio\n      import types\n\n      @types.coroutine\n      def f1():\n        yield from asyncio.sleep(1)\n\n      async def f2():\n        await asyncio.sleep(1)\n\n      @types.coroutine\n      def f3():\n        yield 1\n        yield from asyncio.sleep(1)\n        if __random__:\n          return 1\n        else:\n          return "str"\n\n      async def caller():\n        await f1()\n        await f2()\n        x = await f3()\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      import types\n      from typing import Any, Coroutine, Union\n\n      def caller() -> Coroutine[Any, Any, Union[int, str]]: ...\n      def f1() -> Coroutine[Any, Any, None]: ...\n      def f2() -> Coroutine[Any, Any, None]: ...\n      def f3() -> Coroutine[Any, Any, Union[int, str]]: ...\n    ')

    @test_utils.skipFromPy((3, 11), 'asyncio.coroutine was removed in 3.11')
    def test_asyncio_coroutine_inference(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import asyncio\n      @asyncio.coroutine\n      def f():\n        yield from asyncio.sleep(1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      from typing import Any, Coroutine\n      def f() -> Coroutine[Any, Any, None]: ...\n    ')

    @test_utils.skipBeforePy((3, 11), 'asyncio.coroutine was removed in 3.11')
    def test_asyncio_coroutine_does_not_exist(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import asyncio\n      @asyncio.coroutine  # module-attr\n      def f():\n        yield from asyncio.sleep(1)\n    ')

    def test_generator_based_coroutine_error(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import Generator\n      import types\n\n      @types.coroutine\n      def f1():\n        return 1\n\n      @types.coroutine\n      def f2() -> Generator[int, None, int]:\n        yield 1\n        return 1\n\n      def f3(x, y: str):\n        pass\n\n      async def caller():\n        x = await f1()  # bad-return-type[e1]\n        y = await f2()\n        f3(x, y)  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Awaitable.*int', 'e2': 'y: str.*y: int'})

    def test_awaitable_pyi(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Awaitable, Generator\n      import types\n\n      class BaseAwaitable:\n        def __iter__(self):\n          return self\n\n        __await__ = __iter__\n\n      class SubAwaitable(BaseAwaitable):\n        pass\n\n      async def c1() -> int:\n        return 123\n\n      @types.coroutine\n      def c2() -> Generator[int, None, int]:\n        yield 1\n        return 123\n\n      async def f1():\n        x = await BaseAwaitable()\n        y = await SubAwaitable()\n\n      async def f2(x: Awaitable[int]):\n        return await x\n\n      async def f3():\n        await f2(BaseAwaitable())\n        await f2(SubAwaitable())\n        await f2(c1())\n        await f2(c2())\n    ')
        self.assertTypesMatchPytd(ty, "\n      import types\n      from typing import Any, Awaitable, Coroutine, TypeVar\n\n      _TBaseAwaitable = TypeVar('_TBaseAwaitable', bound=BaseAwaitable)\n\n      class BaseAwaitable:\n          def __await__(self: _TBaseAwaitable) -> _TBaseAwaitable: ...\n          def __iter__(self: _TBaseAwaitable) -> _TBaseAwaitable: ...\n\n      class SubAwaitable(BaseAwaitable):\n          pass\n\n\n      def c1() -> Coroutine[Any, Any, int]: ...\n      def c2() -> Coroutine[Any, Any, int]: ...\n      def f1() -> Coroutine[Any, Any, None]: ...\n      def f2(x: Awaitable[int]) -> Coroutine[Any, Any, int]: ...\n      def f3() -> Coroutine[Any, Any, None]: ...\n    ")

    def test_invalid_awaitable(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class A:\n        pass\n\n      async def fun():\n        await A()  # bad-return-type[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Awaitable.*A'})

    def test_async_for_pyi(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class MyIter:\n        def __aiter__(self):\n          return self\n\n        async def __anext__(self):\n          if __random__:\n            if __random__:\n              return 1\n            else:\n              return "str"\n          else:\n            raise StopAsyncIteration\n\n      async def caller():\n        res = []\n        async for i in MyIter():\n          res.append(i)\n        else:\n          pass\n        return res\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Coroutine, List, TypeVar, Union\n\n      _TMyIter = TypeVar('_TMyIter', bound=MyIter)\n\n      class MyIter:\n          def __aiter__(self: _TMyIter) -> _TMyIter: ...\n          def __anext__(self) -> Coroutine[Any, Any, Union[int, str]]: ...\n\n\n      def caller() -> Coroutine[Any, Any, List[Union[int, str]]]: ...\n    ")

    def test_async_for_error(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Iter1:\n        pass\n\n      class Iter2:\n        def __aiter__(self):\n          return self\n\n      class Iter3:\n        def __aiter__(self):\n          return self\n\n        def __anext__(self):\n          if __random__:\n            if __random__:\n              return 1\n            else:\n              return "str"\n          else:\n            raise StopAsyncIteration\n\n      class Iter4:\n        def __aiter__(self):\n          return self\n\n        async def __anext__(self):\n          if __random__:\n            if __random__:\n              return 1\n            else:\n              return "str"\n          else:\n            raise StopAsyncIteration\n\n      async def caller():\n        res = []\n        async for i in Iter1():  # attribute-error[e1]\n          res.append(i)\n        async for i in Iter2():  # attribute-error[e2]\n          res.append(i)\n        async for i in Iter3():  # bad-return-type[e3]\n          res.append(i)\n        async for i in Iter4():\n          res.append(i)\n        return res\n    ')
        self.assertErrorRegexes(errors, {'e1': 'No attribute.*__aiter__', 'e2': 'No attribute.*__anext__', 'e3': 'Awaitable.*Union\\[int, str\\]'})

    def test_async_with_pyi(self):
        if False:
            return 10
        ty = self.Infer('\n      async def log(s):\n        return s\n\n      class AsyncCtx:\n        async def __aenter__(self):\n          await log("__aenter__")\n          return self\n\n        async def __aexit__(self, exc_type, exc, tb):\n          await log("__aexit__")\n\n        def func():\n          pass\n\n      def fctx(x: AsyncCtx):\n        pass\n\n      async def caller():\n        async with AsyncCtx() as var:\n          var.func()\n          fctx(var)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, Coroutine, TypeVar\n\n      _T0 = TypeVar('_T0')\n\n      class AsyncCtx:\n          def __aenter__(self) -> Coroutine[Any, Any, AsyncCtx]: ...\n          def __aexit__(self, exc_type, exc, tb) -> Coroutine[Any, Any, None]: ...\n          def func() -> None: ...\n\n\n      def caller() -> Coroutine[Any, Any, None]: ...\n      def fctx(x: AsyncCtx) -> None: ...\n      def log(s: _T0) -> Coroutine[Any, Any, _T0]: ...\n    ")

    def test_async_with_error(self):
        if False:
            i = 10
            return i + 15
        if self.python_version >= (3, 10):
            e4 = '  # bad-return-type[e4]'
            e4_pre310 = ''
        else:
            e4 = ''
            e4_pre310 = '  # bad-return-type[e4]'
        errors = self.CheckWithErrors(f'\n      class AsyncCtx1:\n        pass\n\n      class AsyncCtx2:\n        def __aenter__(self):\n          return self\n\n        def __aexit__(self, exc_type, exc, tb):\n          return "str"\n\n      async def caller():\n        ctx1 = AsyncCtx1()\n        ctx2 = AsyncCtx2()\n        async with ctx1 as var1:  # attribute-error[e1]  # attribute-error[e2]\n          async with ctx2 as var2:  # bad-return-type[e3]{e4}\n            pass{e4_pre310}\n    ')
        self.assertErrorRegexes(errors, {'e1': 'No attribute.*__aexit__', 'e2': 'No attribute.*__aenter__', 'e3': 'Awaitable.*AsyncCtx2', 'e4': 'Awaitable.*str'})

    def test_load_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Any, Coroutine, Awaitable, TypeVar\n\n        def f1() -> Coroutine[Any, Any, str]: ...\n        def f2() -> Awaitable[str]: ...\n\n        _TBaseAwaitable = TypeVar('_TBaseAwaitable', bound=BaseAwaitable)\n\n        class BaseAwaitable:\n          def __await__(self: _TBaseAwaitable) -> _TBaseAwaitable: ...\n          def __iter__(self: _TBaseAwaitable) -> _TBaseAwaitable: ...\n\n\n        class SubAwaitable(BaseAwaitable):\n          pass\n\n\n        class MyIter:\n          def __aiter__(self) -> MyIter: ...\n          def __anext__(self) -> Coroutine[Any, Any, str]: ...\n\n\n        class AsyncCtx:\n          def __aenter__(self) -> Coroutine[Any, Any, AsyncCtx]: ...\n          def __aexit__(self, exc_type, exc, tb) -> Coroutine[Any, Any, None]: ...\n          def func() -> None: ...\n      ")
            ty = self.Infer('\n        import foo\n        from typing import Awaitable, Coroutine, Any\n\n        async def func1(x: Awaitable[str]):\n          res = []\n          await foo.BaseAwaitable()\n          await foo.SubAwaitable()\n          res.append(await foo.f1())\n          res.append(await foo.f2())\n          res.append(await x)\n          async for i in foo.MyIter():\n            res.append(i)\n          async with foo.AsyncCtx() as var:\n            var.func()\n          return res\n\n        async def func2(x: Coroutine[Any, Any, str]):\n          res = []\n          await foo.BaseAwaitable()\n          await foo.SubAwaitable()\n          res.append(await foo.f1())\n          res.append(await foo.f2())\n          res.append(await x)\n          async for i in foo.MyIter():\n            res.append(i)\n          async with foo.AsyncCtx() as var:\n            var.func()\n          return res\n\n        func1(foo.f1())\n        func1(foo.f2())\n        func2(foo.f1())\n      ', deep=True, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Awaitable, Coroutine, List\n\n        def func1(x: Awaitable[str]) -> Coroutine[Any, Any, List[str]]: ...\n        def func2(x: Coroutine[Any, Any, str]) -> Coroutine[Any, Any, List[str]]: ...\n      ')

    def test_await_variable_with_multi_bindings(self):
        if False:
            return 10
        ty = self.Infer('\n      async def f1():\n        return 123\n\n      async def f2():\n        return "str"\n\n      async def caller():\n        if __random__:\n          x = f1()\n        else:\n          x = f2()\n        return await x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Coroutine, Union\n\n      def caller() -> Coroutine[Any, Any, Union[int, str]]: ...\n      def f1() -> Coroutine[Any, Any, int]: ...\n      def f2() -> Coroutine[Any, Any, str]: ...\n    ')

    def test_await_generator(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import asyncio\n\n      async def tcp_echo_client(message):\n        return await asyncio.open_connection( '127.0.0.1', 8888)\n    ")
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      from typing import Any, Coroutine, Tuple\n      def tcp_echo_client(message) -> Coroutine[\n        Any, Any,\n        Tuple[asyncio.streams.StreamReader, asyncio.streams.StreamWriter]]: ...\n    ')

    def test_queue(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import asyncio\n\n      async def worker(queue):\n        return await queue.get()\n\n      async def main():\n        queue = asyncio.Queue()\n        worker(queue)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      from typing import Any, Coroutine\n      def worker(queue) -> coroutine: ...\n      def main() -> Coroutine[Any, Any, None]: ...\n    ')

    def test_future(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import asyncio\n\n      async def foo() -> int:\n        return 1\n\n      async def call_foo():\n        for future in asyncio.as_completed([foo()]):\n          return await future\n    ')
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      from typing import Any, Coroutine, Optional\n      def foo() -> Coroutine[Any, Any, int]: ...\n      def call_foo() -> Coroutine[Any, Any, Optional[int]]: ...\n    ')

    def test_pyi_async_def(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        async def f() -> int: ...\n      ')
            ty = self.Infer('\n        import foo\n        c = foo.f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Coroutine\n        c: Coroutine[Any, Any, int]\n      ')
if __name__ == '__main__':
    test_base.main()