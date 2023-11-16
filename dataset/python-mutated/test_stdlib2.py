"""Tests of selected stdlib functions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class StdLibTestsBasic(test_base.BaseTest, test_utils.TestCollectionsMixin):
    """Tests for files in typeshed/stdlib."""

    def test_collections_deque(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Deque\n      import collections\n      def f1(x: Deque): ...\n      def f2(x: int): ...\n      f1(collections.deque())\n      f2(collections.deque())  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*deque'})

    def test_collections_deque_init(self):
        if False:
            return 10
        ty = self.Infer('\n      import collections\n      x = collections.deque([1, 2, 3], maxlen=10)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import collections\n      x = ...  # type: collections.deque[int]\n    ')

    def test_partial(self):
        if False:
            print('Hello World!')
        self.Check("\n      import functools\n      from typing import TypeVar\n      T = TypeVar('T', float, str)\n      def identity(x: T) -> T: return x\n      functools.partial(identity)\n    ")

    def test_collections_container(self):
        if False:
            for i in range(10):
                print('nop')
        self._testCollectionsObject('Container', '[]', '42', 'Container.*int')

    def test_collections_hashable(self):
        if False:
            while True:
                i = 10
        self._testCollectionsObject('Hashable', '42', '[]', 'Hashable.*List')

    def test_collections_iterable(self):
        if False:
            print('Hello World!')
        self._testCollectionsObject('Iterable', '[]', '42', 'Iterable.*int')

    def test_collections_iterator(self):
        if False:
            return 10
        self._testCollectionsObject('Iterator', 'iter([])', '42', 'Iterator.*int')

    def test_collections_sized(self):
        if False:
            return 10
        self._testCollectionsObject('Sized', '[]', '42', 'Sized.*int')

    def test_collections_callable(self):
        if False:
            for i in range(10):
                print('nop')
        self._testCollectionsObject('Callable', 'list', '42', 'Callable.*int')

    def test_collections_sequence(self):
        if False:
            print('Hello World!')
        self._testCollectionsObject('Sequence', '[]', '42', 'Sequence.*int')

    def test_collections_mutable_sequence(self):
        if False:
            i = 10
            return i + 15
        self._testCollectionsObject('MutableSequence', '[]', '42', 'MutableSequence.*int')

    def test_collections_set(self):
        if False:
            return 10
        self._testCollectionsObject('Set', 'set()', '42', 'set.*int')

    def test_collections_mutable_set(self):
        if False:
            i = 10
            return i + 15
        self._testCollectionsObject('MutableSet', 'set()', '42', 'MutableSet.*int')

    def test_collections_mapping(self):
        if False:
            while True:
                i = 10
        self._testCollectionsObject('Mapping', '{}', '42', 'Mapping.*int')

    def test_collections_mutable_mapping(self):
        if False:
            print('Hello World!')
        self._testCollectionsObject('MutableMapping', '{}', '42', 'MutableMapping.*int')

    def test_tempdir_name(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import tempfile\n      def f() -> str:\n        return tempfile.TemporaryDirectory().name\n    ')

    def test_fraction_subclass(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import fractions\n      class MyClass(fractions.Fraction):\n        pass\n      def foo() -> MyClass:\n        return MyClass(1, 2)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import fractions\n      class MyClass(fractions.Fraction): ...\n      def foo() -> MyClass: ...\n  ')

    def test_codetype(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import types\n      class Foo:\n        x: types.CodeType\n        def set_x(self):\n          self.x = compile('', '', '')\n    ")

    def test_os_path_basename(self):
        if False:
            return 10
        self.options.tweak(strict_parameter_checks=False)
        self.Check('\n      import os\n      from typing import Optional\n      x: Optional[str]\n      assert_type(os.path.basename(x), str)\n    ')

    def test_decimal_round(self):
        if False:
            print('Hello World!')
        self.Check("\n      import decimal\n      x = decimal.Decimal('5.02')\n      assert_type(round(x), int)\n      assert_type(round(x, 1), decimal.Decimal)\n    ")

class StdlibTestsFeatures(test_base.BaseTest, test_utils.TestCollectionsMixin):
    """Tests for files in typeshed/stdlib."""

    def test_collections_smoke_test(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import collections\n      collections.AsyncIterable\n      collections.AsyncIterator\n      collections.AsyncGenerator\n      collections.Awaitable\n      collections.Coroutine\n    ')

    def test_collections_bytestring(self):
        if False:
            print('Hello World!')
        self._testCollectionsObject('ByteString', "b'hello'", '42', 'Union\\[bytearray, bytes, memoryview\\].*int')

    def test_collections_collection(self):
        if False:
            return 10
        self._testCollectionsObject('Collection', '[]', '42', 'Collection.*int')

    def test_collections_generator(self):
        if False:
            i = 10
            return i + 15
        self._testCollectionsObject('Generator', 'i for i in range(42)', '42', 'generator.*int')

    def test_collections_reversible(self):
        if False:
            while True:
                i = 10
        self._testCollectionsObject('Reversible', '[]', '42', 'Reversible.*int')

    def test_collections_mapping_view(self):
        if False:
            while True:
                i = 10
        self._testCollectionsObject('MappingView', '{}.items()', '42', 'MappingView.*int')

    def test_collections_items_view(self):
        if False:
            while True:
                i = 10
        self._testCollectionsObject('ItemsView', '{}.items()', '42', 'ItemsView.*int')

    def test_collections_keys_view(self):
        if False:
            i = 10
            return i + 15
        self._testCollectionsObject('KeysView', '{}.keys()', '42', 'KeysView.*int')

    def test_collections_values_view(self):
        if False:
            for i in range(10):
                print('nop')
        self._testCollectionsObject('ValuesView', '{}.values()', '42', 'ValuesView.*int')

    def test_tempfile(self):
        if False:
            return 10
        self.options.tweak(strict_parameter_checks=False)
        ty = self.Infer('\n      import tempfile\n      import typing\n      import os\n      def f(fi: typing.IO):\n        fi.write("foobar")\n        pos = fi.tell()\n        fi.seek(0, os.SEEK_SET)\n        s = fi.read(6)\n        fi.close()\n        return s\n      f(tempfile.TemporaryFile("wb", suffix=".foo"))\n      f(tempfile.NamedTemporaryFile("wb", suffix=".foo"))\n      f(tempfile.SpooledTemporaryFile(1048576, "wb", suffix=".foo"))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import os\n      import tempfile\n      import typing\n      from typing import Any, Union\n      def f(fi: typing.IO) -> Union[bytes, str]: ...\n    ')

    def test_defaultdict(self):
        if False:
            return 10
        self.Check('\n      import collections\n      import itertools\n      ids = collections.defaultdict(itertools.count(17).__next__)\n    ')

    def test_defaultdict_matches_dict(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      from typing import DefaultDict, Dict\n      def take_dict(d: Dict[int, str]): pass\n      def take_defaultdict(d: DefaultDict[int, str]): pass\n      d = collections.defaultdict(str, {1: "hello"})\n      take_dict(d)\n      take_defaultdict(d)\n    ')

    def test_defaultdict_kwargs(self):
        if False:
            return 10
        self.Check("\n      import collections\n      from typing import DefaultDict, Union\n      def take_str_int_values(d: DefaultDict[str, Union[str, int]]): pass\n      d = collections.defaultdict(str, {'x': 'x'}, an_int = 1)\n      take_str_int_values(d)\n      def take_three_types(d: DefaultDict[str, Union[str, int, list]]): pass\n      e = collections.defaultdict(str, {'x': [1, 2]}, an_int = 3)\n      take_three_types(e)\n      collections.defaultdict(None, [(1, '2'), (3, '4')], a=1, b=2)\n    ")

    def test_sys_version_info_lt(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import sys\n      if sys.version_info[0] < 3:\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v = ...  # type: str\n    ')

    def test_sys_version_info_le(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import sys\n      if sys.version_info[0] <= 3:\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v = ...  # type: int\n    ')

    def test_sys_version_info_eq(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import sys\n      if sys.version_info[0] == 2:\n        v = 42\n      elif sys.version_info[0] == 3:\n        v = "hello world"\n      else:\n        v = None\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v = ...  # type: str\n    ')

    def test_sys_version_info_ge(self):
        if False:
            return 10
        ty = self.Infer('\n      import sys\n      if sys.version_info[0] >= 3:\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v = ...  # type: int\n    ')

    def test_sys_version_info_gt(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import sys\n      if sys.version_info[0] > 2:\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v = ...  # type: int\n    ')

    def test_sys_version_info_named_attribute(self):
        if False:
            return 10
        ty = self.Infer('\n      import sys\n      if sys.version_info.major == 2:\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v: str\n    ')

    def test_sys_version_info_tuple(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import sys\n      if sys.version_info >= (3, 5):\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v: int\n    ')

    def test_sys_version_info_slice(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import sys\n      if sys.version_info[:2] >= (3, 5):\n        v = 42\n      else:\n        v = "hello world"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      v: int\n    ')

    def test_sys_platform(self):
        if False:
            print('Hello World!')
        self.options.tweak(platform='linux')
        ty = self.Infer('\n      import sys\n      if sys.platform == "linux":\n        x = 0\n      else:\n        x = "0"\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      x: int\n    ')

    def test_async(self):
        if False:
            i = 10
            return i + 15
        'Test various asyncio features.'
        ty = self.Infer('\n      import asyncio\n      async def log(x: str):\n        return x\n      class AsyncContextManager:\n        async def __aenter__(self):\n          await log("entering context")\n        async def __aexit__(self, exc_type, exc, tb):\n          await log("exiting context")\n      async def my_coroutine(seconds_to_sleep=0.4):\n          await asyncio.sleep(seconds_to_sleep)\n      async def test_with(x):\n        try:\n          async with x as y:\n            pass\n        finally:\n          pass\n      event_loop = asyncio.get_event_loop()\n      try:\n        event_loop.run_until_complete(my_coroutine())\n      finally:\n        event_loop.close()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import asyncio\n      from typing import Any, Coroutine\n\n      event_loop: asyncio.events.AbstractEventLoop\n\n      class AsyncContextManager:\n          def __aenter__(self) -> Coroutine[Any, Any, None]: ...\n          def __aexit__(self, exc_type, exc, tb) -> Coroutine[Any, Any, None]: ...\n      def log(x: str) -> Coroutine[Any, Any, str]: ...\n      def my_coroutine(seconds_to_sleep = ...) -> Coroutine[Any, Any, None]: ...\n      def test_with(x) -> Coroutine[Any, Any, None]: ...\n    ')

    def test_async_iter(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import asyncio\n      class AsyncIterable:\n        def __aiter__(self):\n          return self\n        async def __anext__(self):\n          data = await self.fetch_data()\n          if data:\n            return data\n          else:\n            raise StopAsyncIteration\n        async def fetch_data(self):\n          return 1\n      async def iterate(x):\n        async for i in x:\n          pass\n        else:\n          pass\n      iterate(AsyncIterable())\n    ')
        self.assertTypesMatchPytd(ty, "\n      import asyncio\n      from typing import Any, Coroutine, TypeVar\n      _TAsyncIterable = TypeVar('_TAsyncIterable', bound=AsyncIterable)\n      class AsyncIterable:\n          def __aiter__(self: _TAsyncIterable) -> _TAsyncIterable: ...\n          def __anext__(self) -> Coroutine[Any, Any, int]: ...\n          def fetch_data(self) -> Coroutine[Any, Any, int]: ...\n      def iterate(x) -> Coroutine[Any, Any, None]: ...\n    ")

    def test_subprocess(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import subprocess\n      subprocess.run\n    ')

    def test_popen_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)\n        stdout, _ = proc.communicate()\n        return stdout\n    ')
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      def run(cmd) -> bytes: ...\n    ')

    def test_popen_bytes_no_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(cmd, encoding=None, stdout=subprocess.PIPE)\n        stdout, _ = proc.communicate()\n        return stdout\n    ')
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      def run(cmd) -> bytes: ...\n    ')

    def test_popen_bytes_no_universal_newlines(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(\n            cmd, universal_newlines=False, stdout=subprocess.PIPE)\n        stdout, _ = proc.communicate()\n        return stdout\n    ')
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      def run(cmd) -> bytes: ...\n    ')

    def test_popen_str_encoding(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(cmd, encoding='utf-8', stdout=subprocess.PIPE)\n        stdout, _ = proc.communicate()\n        return stdout\n    ")
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      def run(cmd) -> str: ...\n    ')

    def test_popen_str_universal_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import subprocess\n      def run(cmd):\n        proc = subprocess.Popen(\n            cmd, universal_newlines=True, stdout=subprocess.PIPE)\n        stdout, _ = proc.communicate()\n        return stdout\n    ')
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      def run(cmd) -> str: ...\n    ')

    def test_popen_ambiguous_universal_newlines(self):
        if False:
            return 10
        ty = self.Infer("\n      import subprocess\n      from typing import Any\n      def run1(value: bool):\n        proc = subprocess.Popen(['ls'], universal_newlines=value)\n        stdout, _ = proc.communicate()\n        return stdout\n      def run2(value: Any):\n        proc = subprocess.Popen(['ls'], universal_newlines=value)\n        stdout, _ = proc.communicate()\n        return stdout\n    ")
        self.assertTypesMatchPytd(ty, '\n      import subprocess\n      from typing import Any\n      def run1(value: bool) -> Any: ...\n      def run2(value: Any) -> Any: ...\n    ')

    def test_popen_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import subprocess\n      def popen(cmd: str, **kwargs):\n        kwargs['stdout'] = subprocess.PIPE\n        kwargs['stderr'] = subprocess.PIPE\n        process = subprocess.Popen(cmd, **kwargs)\n        stdout, _ = process.communicate()\n        assert_type(stdout, 'Any')\n    ")

    def test_enum(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import enum\n      class Foo(enum.Enum):\n        foo = 0\n        bar = enum.auto()\n      def f(x: Foo):\n        pass\n      f(Foo.foo)\n    ')

    def test_contextlib(self):
        if False:
            while True:
                i = 10
        self.Check('from contextlib import AbstractContextManager')

    def test_chainmap(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import collections\n      v1 = collections.ChainMap({'a': 'b'}, {b'c': 0})\n      v2 = v1.maps\n      v3 = v1.parents\n      v4 = v1.new_child()\n    ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import ChainMap, List, MutableMapping, Union\n      v1: ChainMap[Union[bytes, str], Union[int, str]]\n      v2: List[MutableMapping[Union[bytes, str], Union[int, str]]]\n      v3: ChainMap[Union[bytes, str], Union[int, str]]\n      v4: ChainMap[Union[bytes, str], Union[int, str]]\n    ')

    def test_re(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import re\n      pattern = re.compile('')\n      match = pattern.fullmatch('')\n      if match:\n        group = match[0]\n    ")
        self.assertTypesMatchPytd(ty, '\n      import re\n      from typing import Optional\n      pattern: re.Pattern[str]\n      match: Optional[re.Match[str]]\n      group: str\n    ')

    def test_textio_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import sys\n      sys.stdout.buffer\n    ')

    def test_io_open(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import io\n      def f(name):\n        return io.open(name, "rb").read()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import io\n      def f(name) -> bytes: ...\n    ')

    def test_array_frombytes(self):
        if False:
            print('Hello World!')
        self.Check('\n      import array\n      def f(x: array.array, y: bytes):\n        return x.frombytes(y)\n    ')

    def test_property_attributes(self):
        if False:
            return 10
        self.Check('\n      class C:\n        @property\n        def x(self):\n          pass\n      print(C.x.fget, C.x.fset, C.x.fdel)\n    ')

    def test_re_and_typing(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      import re\n      from typing import Match, Optional, Pattern\n\n      ok1: Pattern = re.compile("")\n      ok2: Optional[Match] = re.match("", "")\n\n      no1: Pattern = 0  # annotation-type-mismatch\n      no2: Match = 0  # annotation-type-mismatch\n    ')

    def test_contextmanager_keywordonly(self):
        if False:
            return 10
        ty = self.Infer('\n      from contextlib import contextmanager\n      @contextmanager\n      def myctx(*, msg=None):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Callable, Iterator, ParamSpec, TypeVar\n      _P = ParamSpec('_P')\n      _T_co = TypeVar('_T_co')\n      def contextmanager(\n          func: Callable[_P, Iterator[_T_co]]\n      ) -> Callable[_P, contextlib._GeneratorContextManager[_T_co]]: ...\n      def myctx(*, msg = ...) -> contextlib._GeneratorContextManager: ...\n    ")
if __name__ == '__main__':
    test_base.main()