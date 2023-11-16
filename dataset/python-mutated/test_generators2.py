"""Tests for generators."""
from pytype.tests import test_base
from pytype.tests import test_utils

class GeneratorBasicTest(test_base.BaseTest):
    """Tests for iterators, generators, coroutines, and yield."""

    def test_return_before_yield(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Generator\n      def f() -> generator:\n        if __random__:\n          return\n        yield 5\n    ')

    def test_empty_iterator(self):
        if False:
            return 10
        self.Check('\n      from typing import Iterator\n      def f() -> Iterator:\n        yield 5\n    ')

    def test_empty_iterable(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Iterable\n      def f() -> Iterable:\n        yield 5\n    ')

    def test_no_return(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import Generator\n      def f() -> Generator[str, None, None]:\n        yield 42  # bad-return-type[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'str.*int'})

class GeneratorFeatureTest(test_base.BaseTest):
    """Tests for iterators, generators, coroutines, and yield."""

    def test_yield_ret_type(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Generator\n      def f(x):\n        if x == 1:\n          yield 1\n          return 1\n        else:\n          yield "1"\n          return "1"\n\n      x = f(2)\n      y = f(1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator, Union\n      def f(x) -> Generator[Union[int, str], Any, Union[int, str]]: ...\n      x = ...  # type: Generator[str, Any, str]\n      y = ...  # type: Generator[int, Any, int]\n    ')

    def test_yield_type_infer(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def gen():\n        l = [1, 2, 3]\n        for x in l:\n          yield x\n        x = "str"\n        yield x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator, Union\n\n      def gen() -> Generator[Union[int, str], Any, None]: ...\n    ')

    def test_send_ret_type(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Generator, Any\n      def f() -> Generator[str, int, Any]:\n        x = yield "5"\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def f() -> Generator[str, int, Any]: ...\n    ')

    def test_parameter_count(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import Generator\n\n      def func1() -> Generator[int, int, int]:\n        x = yield 5\n        return x\n\n      def func2() -> Generator[int, int]:  # invalid-annotation[e1]\n        x = yield 5\n\n      def func3() -> Generator[int]:  # invalid-annotation[e2]\n        yield 5\n    ')
        self.assertErrorSequences(errors, {'e1': ['generator[int, int]', 'generator[_T, _T2, _V]', '3', '2'], 'e2': ['generator[int]', 'generator[_T, _T2, _V]', '3', '1']})

    def test_hidden_fields(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Generator\n      from types import GeneratorType\n      a: generator = __any_object__\n      a.gi_code\n      a.gi_frame\n      a.gi_running\n      a.gi_yieldfrom\n\n      b: Generator = __any_object__\n      b.gi_code\n      b.gi_frame\n      b.gi_running\n      b.gi_yieldfrom\n\n      c: GeneratorType = __any_object__\n      c.gi_code\n      c.gi_frame\n      c.gi_running\n      c.gi_yieldfrom\n    ')

    def test_empty_yield_from(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import abc\n        from typing import Any, AsyncContextManager, Coroutine\n        class Connection(AsyncContextManager): ...\n        class ConnectionFactory(metaclass=abc.ABCMeta):\n          @abc.abstractmethod\n          def new(self) -> Coroutine[Any, Any, Connection]: ...\n      ')
            self.Check('\n        from typing import Any\n        from foo import ConnectionFactory\n        class RetryingConnection:\n          _connection_factory: ConnectionFactory\n          _reinitializer: Any\n          async def _run_loop(self):\n            conn_fut = self._connection_factory.new()\n            async with (await conn_fut) as connection:\n              await connection\n      ', pythonpath=[d.path])

    def test_yield_from(self):
        if False:
            return 10
        ty = self.Infer("\n      def foo():\n        yield 'hello'\n      def bar():\n        yield from foo()\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      def foo() -> Generator[str, Any, None]: ...\n      def bar() -> Generator[str, Any, None]: ...\n    ')

    def test_yield_from_check_return(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Generator\n      def foo():\n        yield 'hello'\n      def bar() -> Generator[str, None, None]:\n        yield from foo()\n      def baz() -> Generator[int, None, None]:\n        yield from foo()  # bad-return-type\n    ")
if __name__ == '__main__':
    test_base.main()