"""Tests for async generators."""
from pytype.tests import test_base

class AsyncGeneratorFeatureTest(test_base.BaseTest):
    """Tests for async iterable, iterator, context manager, generator."""

    def test_empty_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import AsyncIterator, AsyncIterable, AsyncGenerator\n      async def f() -> AsyncIterator:\n        yield 5\n\n      async def f() -> AsyncIterable:\n        yield 5\n\n      async def f() -> AsyncGenerator:\n        yield 5\n    ')

    def test_union_annotation(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import AsyncGenerator, AsyncIterator, AsyncIterable, Union\n\n      async def f() -> Union[AsyncGenerator, AsyncIterator, AsyncIterable]:\n        yield 5\n    ')

    def test_annotation_with_type(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import AsyncGenerator, AsyncIterator, AsyncIterable\n\n      async def gen1() -> AsyncGenerator[int, int]:\n        yield 1\n\n      async def gen2() -> AsyncIterator[int]:\n        yield 1\n\n      async def gen3() -> AsyncIterable[int]:\n        yield 1\n    ')

    def test_yield_type_infer(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      async def f(x):\n        if x == 1:\n          yield 1\n        else:\n          yield "1"\n\n      x = f(2)\n      y = f(1)\n\n      async def func():\n        return "str"\n\n      async def gen():\n        l = [1, 2, 3]\n        for x in l:\n          yield x\n        x = await func()\n        yield x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, AsyncGenerator, Coroutine, Union\n\n      x: AsyncGenerator[str, Any]\n      y: AsyncGenerator[int, Any]\n\n      def f(x) -> AsyncGenerator[Union[int, str], Any]: ...\n      def func() -> Coroutine[Any, Any, str]: ...\n      def gen() -> AsyncGenerator[Union[int, str], Any]: ...\n    ')

    def test_annotation_error(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import AsyncGenerator, AsyncIterator, AsyncIterable, Any, Union\n\n      async def gen1() -> AsyncGenerator[bool, int]:\n        yield 1  # bad-return-type[e1]\n\n      async def gen2() -> AsyncIterator[bool]:\n        yield 1  # bad-return-type[e2]\n\n      async def gen3() -> AsyncIterable[bool]:\n        yield 1  # bad-return-type[e3]\n\n      async def gen4() -> int:  # bad-yield-annotation[e4]\n        yield 1\n\n      async def fun():\n        g = gen1()\n        await g.asend("str")  # wrong-arg-types[e5]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'bool.*int', 'e2': 'bool.*int', 'e3': 'bool.*int', 'e4': 'AsyncGenerator.*AsyncIterable.*AsyncIterator', 'e5': 'int.*str'})

    def test_match_base_class_error(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing import AsyncGenerator, AsyncIterator, AsyncIterable, Union, Any\n\n      async def func():\n        return "str"\n\n      async def gen() -> AsyncGenerator[Union[int, str], Any]:\n        l = [1, 2, 3]\n        for x in l:\n          yield x\n        x = await func()\n        yield x\n\n      def f1(x: AsyncIterator[Union[int, str]]):\n        pass\n\n      def f2(x: AsyncIterator[bool]):\n        pass\n\n      def f3(x: AsyncIterable[Union[int, str]]):\n        pass\n\n      def f4(x: AsyncIterable[bool]):\n        pass\n\n      def f5(x: AsyncGenerator[Union[int, str], Any]):\n        pass\n\n      def f6(x: AsyncGenerator[bool, Any]):\n        pass\n\n      f1(gen())\n      f2(gen())  # wrong-arg-types[e1]\n      f3(gen())\n      f4(gen())  # wrong-arg-types[e2]\n      f5(gen())\n      f6(gen())  # wrong-arg-types[e3]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'bool.*Union\\[int, str\\]', 'e2': 'bool.*Union\\[int, str\\]', 'e3': 'bool.*Union\\[int, str\\]'})

    def test_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import AsyncIterator, AsyncIterable, AsyncGenerator, AsyncContextManager, Any\n\n      async def func():\n        return "str"\n\n      class AIterable:\n        def __aiter__(self):\n          return self\n\n      class AIterator:\n        def __aiter__(self):\n          return self\n\n        async def __anext__(self):\n          if __random__:\n            return 5\n          raise StopAsyncIteration\n\n      class ACtxMgr:\n        async def __aenter__(self):\n          return 5\n\n        async def __aexit__(self, exc_type, exc_value, traceback):\n          pass\n\n      def f1(x: AsyncIterator):\n        pass\n\n      def f2(x: AsyncIterable):\n        pass\n\n      def f3(x: AsyncContextManager):\n        pass\n\n      async def f4():\n        f1(AIterator())\n        f2(AIterable())\n        f3(ACtxMgr())\n        async with ACtxMgr() as x:\n          await func()\n        return x\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Any, AsyncIterable, AsyncIterator, AsyncContextManager, Coroutine, TypeVar\n\n      _TAIterable = TypeVar('_TAIterable', bound=AIterable)\n      _TAIterator = TypeVar('_TAIterator', bound=AIterator)\n\n      class ACtxMgr:\n          def __aenter__(self) -> Coroutine[Any, Any, int]: ...\n          def __aexit__(self, exc_type, exc_value, traceback) -> Coroutine[Any, Any, None]: ...\n\n      class AIterable:\n          def __aiter__(self: _TAIterable) -> _TAIterable: ...\n\n      class AIterator:\n          def __aiter__(self: _TAIterator) -> _TAIterator: ...\n          def __anext__(self) -> Coroutine[Any, Any, int]: ...\n\n\n      def f1(x: AsyncIterator) -> None: ...\n      def f2(x: AsyncIterable) -> None: ...\n      def f3(x: AsyncContextManager) -> None: ...\n      def f4() -> Coroutine[Any, Any, int]: ...\n      def func() -> Coroutine[Any, Any, str]: ...\n    ")
if __name__ == '__main__':
    test_base.main()