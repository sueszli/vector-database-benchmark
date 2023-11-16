from __future__ import annotations
import contextlib
import inspect
import io
import os
import re
import sys
import tempfile
import textwrap
import unittest
from typing import Any, Callable, Iterator
import mypy.stubtest
from mypy.stubtest import parse_options, test_stubs
from mypy.test.data import root_dir

@contextlib.contextmanager
def use_tmp_dir(mod_name: str) -> Iterator[str]:
    if False:
        while True:
            i = 10
    current = os.getcwd()
    current_syspath = sys.path.copy()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            if sys.path[0] != tmp:
                sys.path.insert(0, tmp)
            yield tmp
        finally:
            sys.path = current_syspath.copy()
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            os.chdir(current)
TEST_MODULE_NAME = 'test_module'
stubtest_typing_stub = '\nAny = object()\n\nclass _SpecialForm:\n    def __getitem__(self, typeargs: Any) -> object: ...\n\nCallable: _SpecialForm = ...\nGeneric: _SpecialForm = ...\nProtocol: _SpecialForm = ...\nUnion: _SpecialForm = ...\n\nclass TypeVar:\n    def __init__(self, name, covariant: bool = ..., contravariant: bool = ...) -> None: ...\n\nclass ParamSpec:\n    def __init__(self, name: str) -> None: ...\n\nAnyStr = TypeVar("AnyStr", str, bytes)\n_T = TypeVar("_T")\n_T_co = TypeVar("_T_co", covariant=True)\n_K = TypeVar("_K")\n_V = TypeVar("_V")\n_S = TypeVar("_S", contravariant=True)\n_R = TypeVar("_R", covariant=True)\n\nclass Coroutine(Generic[_T_co, _S, _R]): ...\nclass Iterable(Generic[_T_co]): ...\nclass Iterator(Iterable[_T_co]): ...\nclass Mapping(Generic[_K, _V]): ...\nclass Match(Generic[AnyStr]): ...\nclass Sequence(Iterable[_T_co]): ...\nclass Tuple(Sequence[_T_co]): ...\nclass NamedTuple(tuple[Any, ...]): ...\ndef overload(func: _T) -> _T: ...\ndef type_check_only(func: _T) -> _T: ...\ndef deprecated(__msg: str) -> Callable[[_T], _T]: ...\ndef final(func: _T) -> _T: ...\n'
stubtest_builtins_stub = "\nfrom typing import Generic, Mapping, Sequence, TypeVar, overload\n\nT = TypeVar('T')\nT_co = TypeVar('T_co', covariant=True)\nKT = TypeVar('KT')\nVT = TypeVar('VT')\n\nclass object:\n    __module__: str\n    def __init__(self) -> None: pass\n    def __repr__(self) -> str: pass\nclass type: ...\n\nclass tuple(Sequence[T_co], Generic[T_co]):\n    def __ge__(self, __other: tuple[T_co, ...]) -> bool: pass\n\nclass dict(Mapping[KT, VT]): ...\n\nclass function: pass\nclass ellipsis: pass\n\nclass int: ...\nclass float: ...\nclass bool(int): ...\nclass str: ...\nclass bytes: ...\n\nclass list(Sequence[T]): ...\n\ndef property(f: T) -> T: ...\ndef classmethod(f: T) -> T: ...\ndef staticmethod(f: T) -> T: ...\n"
stubtest_enum_stub = "\nimport sys\nfrom typing import Any, TypeVar, Iterator\n\n_T = TypeVar('_T')\n\nclass EnumMeta(type):\n    def __len__(self) -> int: pass\n    def __iter__(self: type[_T]) -> Iterator[_T]: pass\n    def __reversed__(self: type[_T]) -> Iterator[_T]: pass\n    def __getitem__(self: type[_T], name: str) -> _T: pass\n\nclass Enum(metaclass=EnumMeta):\n    def __new__(cls: type[_T], value: object) -> _T: pass\n    def __repr__(self) -> str: pass\n    def __str__(self) -> str: pass\n    def __format__(self, format_spec: str) -> str: pass\n    def __hash__(self) -> Any: pass\n    def __reduce_ex__(self, proto: Any) -> Any: pass\n    name: str\n    value: Any\n\nclass Flag(Enum):\n    def __or__(self: _T, other: _T) -> _T: pass\n    def __and__(self: _T, other: _T) -> _T: pass\n    def __xor__(self: _T, other: _T) -> _T: pass\n    def __invert__(self: _T) -> _T: pass\n    if sys.version_info >= (3, 11):\n        __ror__ = __or__\n        __rand__ = __and__\n        __rxor__ = __xor__\n"

def run_stubtest(stub: str, runtime: str, options: list[str], config_file: str | None=None) -> str:
    if False:
        i = 10
        return i + 15
    with use_tmp_dir(TEST_MODULE_NAME) as tmp_dir:
        with open('builtins.pyi', 'w') as f:
            f.write(stubtest_builtins_stub)
        with open('typing.pyi', 'w') as f:
            f.write(stubtest_typing_stub)
        with open('enum.pyi', 'w') as f:
            f.write(stubtest_enum_stub)
        with open(f'{TEST_MODULE_NAME}.pyi', 'w') as f:
            f.write(stub)
        with open(f'{TEST_MODULE_NAME}.py', 'w') as f:
            f.write(runtime)
        if config_file:
            with open(f'{TEST_MODULE_NAME}_config.ini', 'w') as f:
                f.write(config_file)
            options = options + ['--mypy-config-file', f'{TEST_MODULE_NAME}_config.ini']
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            test_stubs(parse_options([TEST_MODULE_NAME] + options), use_builtins_fixtures=True)
        return remove_color_code(output.getvalue().replace(os.path.realpath(tmp_dir) + os.sep, '').replace(tmp_dir + os.sep, ''))

class Case:

    def __init__(self, stub: str, runtime: str, error: str | None):
        if False:
            return 10
        self.stub = stub
        self.runtime = runtime
        self.error = error

def collect_cases(fn: Callable[..., Iterator[Case]]) -> Callable[..., None]:
    if False:
        print('Hello World!')
    "run_stubtest used to be slow, so we used this decorator to combine cases.\n\n    If you're reading this and bored, feel free to refactor this and make it more like\n    other mypy tests.\n\n    "

    def test(*args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        cases = list(fn(*args, **kwargs))
        expected_errors = set()
        for c in cases:
            if c.error is None:
                continue
            expected_error = c.error
            if expected_error == '':
                expected_error = TEST_MODULE_NAME
            elif not expected_error.startswith(f'{TEST_MODULE_NAME}.'):
                expected_error = f'{TEST_MODULE_NAME}.{expected_error}'
            assert expected_error not in expected_errors, 'collect_cases merges cases into a single stubtest invocation; we already expect an error for {}'.format(expected_error)
            expected_errors.add(expected_error)
        output = run_stubtest(stub='\n\n'.join((textwrap.dedent(c.stub.lstrip('\n')) for c in cases)), runtime='\n\n'.join((textwrap.dedent(c.runtime.lstrip('\n')) for c in cases)), options=['--generate-allowlist'])
        actual_errors = set(output.splitlines())
        assert actual_errors == expected_errors, output
    return test

class StubtestUnit(unittest.TestCase):

    @collect_cases
    def test_basic_good(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='def f(number: int, text: str) -> None: ...', runtime='def f(number, text): pass', error=None)
        yield Case(stub='\n            class X:\n                def f(self, number: int, text: str) -> None: ...\n            ', runtime='\n            class X:\n                def f(self, number, text): pass\n            ', error=None)

    @collect_cases
    def test_types(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='def mistyped_class() -> None: ...', runtime='class mistyped_class: pass', error='mistyped_class')
        yield Case(stub='class mistyped_fn: ...', runtime='def mistyped_fn(): pass', error='mistyped_fn')
        yield Case(stub='\n            class X:\n                def mistyped_var(self) -> int: ...\n            ', runtime='\n            class X:\n                mistyped_var = 1\n            ', error='X.mistyped_var')

    @collect_cases
    def test_coroutines(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='def bar() -> int: ...', runtime='async def bar(): return 5', error='bar')
        yield Case(stub='async def foo() -> int: ...', runtime='def foo(): return 5', error=None)
        yield Case(stub='def baz() -> int: ...', runtime='def baz(): return 5', error=None)
        yield Case(stub='async def bingo() -> int: ...', runtime='async def bingo(): return 5', error=None)

    @collect_cases
    def test_arg_name(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='def bad(number: int, text: str) -> None: ...', runtime='def bad(num, text) -> None: pass', error='bad')
        yield Case(stub='def good_posonly(__number: int, text: str) -> None: ...', runtime='def good_posonly(num, /, text): pass', error=None)
        yield Case(stub='def bad_posonly(__number: int, text: str) -> None: ...', runtime='def bad_posonly(flag, /, text): pass', error='bad_posonly')
        yield Case(stub='\n            class BadMethod:\n                def f(self, number: int, text: str) -> None: ...\n            ', runtime='\n            class BadMethod:\n                def f(self, n, text): pass\n            ', error='BadMethod.f')
        yield Case(stub='\n            class GoodDunder:\n                def __exit__(self, t, v, tb) -> None: ...\n            ', runtime='\n            class GoodDunder:\n                def __exit__(self, exc_type, exc_val, exc_tb): pass\n            ', error=None)

    @collect_cases
    def test_arg_kind(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='def runtime_kwonly(number: int, text: str) -> None: ...', runtime='def runtime_kwonly(number, *, text): pass', error='runtime_kwonly')
        yield Case(stub='def stub_kwonly(number: int, *, text: str) -> None: ...', runtime='def stub_kwonly(number, text): pass', error='stub_kwonly')
        yield Case(stub='def stub_posonly(__number: int, text: str) -> None: ...', runtime='def stub_posonly(number, text): pass', error='stub_posonly')
        yield Case(stub='def good_posonly(__number: int, text: str) -> None: ...', runtime='def good_posonly(number, /, text): pass', error=None)
        yield Case(stub='def runtime_posonly(number: int, text: str) -> None: ...', runtime='def runtime_posonly(number, /, text): pass', error='runtime_posonly')
        yield Case(stub='def stub_posonly_570(number: int, /, text: str) -> None: ...', runtime='def stub_posonly_570(number, text): pass', error='stub_posonly_570')

    @collect_cases
    def test_default_presence(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='def f1(text: str = ...) -> None: ...', runtime="def f1(text = 'asdf'): pass", error=None)
        yield Case(stub='def f2(text: str = ...) -> None: ...', runtime='def f2(text): pass', error='f2')
        yield Case(stub='def f3(text: str) -> None: ...', runtime="def f3(text = 'asdf'): pass", error='f3')
        yield Case(stub='def f4(text: str = ...) -> None: ...', runtime='def f4(text = None): pass', error='f4')
        yield Case(stub='def f5(data: bytes = ...) -> None: ...', runtime="def f5(data = 'asdf'): pass", error='f5')
        yield Case(stub='\n            from typing import TypeVar\n            _T = TypeVar("_T", bound=str)\n            def f6(text: _T = ...) -> None: ...\n            ', runtime='def f6(text = None): pass', error='f6')

    @collect_cases
    def test_default_value(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub="def f1(text: str = 'x') -> None: ...", runtime="def f1(text = 'y'): pass", error='f1')
        yield Case(stub='def f2(text: bytes = b"x\'") -> None: ...', runtime='def f2(text = b"x\'"): pass', error=None)
        yield Case(stub='def f3(text: bytes = b"y\'") -> None: ...', runtime='def f3(text = b"x\'"): pass', error='f3')
        yield Case(stub='def f4(text: object = 1) -> None: ...', runtime='def f4(text = 1.0): pass', error='f4')
        yield Case(stub='def f5(text: object = True) -> None: ...', runtime='def f5(text = 1): pass', error='f5')
        yield Case(stub='def f6(text: object = True) -> None: ...', runtime='def f6(text = True): pass', error=None)
        yield Case(stub='def f7(text: object = not True) -> None: ...', runtime='def f7(text = False): pass', error=None)
        yield Case(stub='def f8(text: object = not True) -> None: ...', runtime='def f8(text = True): pass', error='f8')
        yield Case(stub='def f9(text: object = {1: 2}) -> None: ...', runtime='def f9(text = {1: 3}): pass', error='f9')
        yield Case(stub='def f10(text: object = [1, 2]) -> None: ...', runtime='def f10(text = [1, 2]): pass', error=None)

    @collect_cases
    def test_static_class_method(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='\n            class Good:\n                @classmethod\n                def f(cls, number: int, text: str) -> None: ...\n            ', runtime='\n            class Good:\n                @classmethod\n                def f(cls, number, text): pass\n            ', error=None)
        yield Case(stub='\n            class Bad1:\n                def f(cls, number: int, text: str) -> None: ...\n            ', runtime='\n            class Bad1:\n                @classmethod\n                def f(cls, number, text): pass\n            ', error='Bad1.f')
        yield Case(stub='\n            class Bad2:\n                @classmethod\n                def f(cls, number: int, text: str) -> None: ...\n            ', runtime='\n            class Bad2:\n                @staticmethod\n                def f(self, number, text): pass\n            ', error='Bad2.f')
        yield Case(stub='\n            class Bad3:\n                @staticmethod\n                def f(cls, number: int, text: str) -> None: ...\n            ', runtime='\n            class Bad3:\n                @classmethod\n                def f(self, number, text): pass\n            ', error='Bad3.f')
        yield Case(stub='\n            class GoodNew:\n                def __new__(cls, *args, **kwargs): ...\n            ', runtime='\n            class GoodNew:\n                def __new__(cls, *args, **kwargs): pass\n            ', error=None)

    @collect_cases
    def test_arg_mismatch(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='def f1(a, *, b, c) -> None: ...', runtime='def f1(a, *, b, c): pass', error=None)
        yield Case(stub='def f2(a, *, b) -> None: ...', runtime='def f2(a, *, b, c): pass', error='f2')
        yield Case(stub='def f3(a, *, b, c) -> None: ...', runtime='def f3(a, *, b): pass', error='f3')
        yield Case(stub='def f4(a, *, b, c) -> None: ...', runtime='def f4(a, b, *, c): pass', error='f4')
        yield Case(stub='def f5(a, b, *, c) -> None: ...', runtime='def f5(a, *, b, c): pass', error='f5')

    @collect_cases
    def test_varargs_varkwargs(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='def f1(*args, **kwargs) -> None: ...', runtime='def f1(*args, **kwargs): pass', error=None)
        yield Case(stub='def f2(*args, **kwargs) -> None: ...', runtime='def f2(**kwargs): pass', error='f2')
        yield Case(stub='def g1(a, b, c, d) -> None: ...', runtime='def g1(a, *args): pass', error=None)
        yield Case(stub='def g2(a, b, c, d, *args) -> None: ...', runtime='def g2(a): pass', error='g2')
        yield Case(stub='def g3(a, b, c, d, *args) -> None: ...', runtime='def g3(a, *args): pass', error=None)
        yield Case(stub='def h1(a) -> None: ...', runtime='def h1(a, b, c, d, *args): pass', error='h1')
        yield Case(stub='def h2(a, *args) -> None: ...', runtime='def h2(a, b, c, d): pass', error='h2')
        yield Case(stub='def h3(a, *args) -> None: ...', runtime='def h3(a, b, c, d, *args): pass', error='h3')
        yield Case(stub='def j1(a: int, *args) -> None: ...', runtime='def j1(a): pass', error='j1')
        yield Case(stub='def j2(a: int) -> None: ...', runtime='def j2(a, *args): pass', error='j2')
        yield Case(stub='def j3(a, b, c) -> None: ...', runtime='def j3(a, *args, c): pass', error='j3')
        yield Case(stub='def k1(a, **kwargs) -> None: ...', runtime='def k1(a): pass', error='k1')
        yield Case(stub='def k2(a) -> None: ...', runtime='def k2(a, **kwargs): pass', error=None)
        yield Case(stub='def k3(a, b) -> None: ...', runtime='def k3(a, **kwargs): pass', error='k3')
        yield Case(stub='def k4(a, *, b) -> None: ...', runtime='def k4(a, **kwargs): pass', error=None)
        yield Case(stub='def k5(a, *, b) -> None: ...', runtime='def k5(a, *, b, c, **kwargs): pass', error='k5')
        yield Case(stub='def k6(a, *, b, **kwargs) -> None: ...', runtime='def k6(a, *, b, c, **kwargs): pass', error='k6')

    @collect_cases
    def test_overload(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='\n            from typing import overload\n\n            @overload\n            def f1(a: int, *, c: int = ...) -> int: ...\n            @overload\n            def f1(a: int, b: int, c: int = ...) -> str: ...\n            ', runtime='def f1(a, b = 0, c = 0): pass', error=None)
        yield Case(stub='\n            @overload\n            def f2(a: int, *, c: int = ...) -> int: ...\n            @overload\n            def f2(a: int, b: int, c: int = ...) -> str: ...\n            ', runtime='def f2(a, b, c = 0): pass', error='f2')
        yield Case(stub='\n            @overload\n            def f3(a: int) -> int: ...\n            @overload\n            def f3(a: int, b: str) -> str: ...\n            ', runtime='def f3(a, b = None): pass', error='f3')
        yield Case(stub='\n            @overload\n            def f4(a: int, *args, b: int, **kwargs) -> int: ...\n            @overload\n            def f4(a: str, *args, b: int, **kwargs) -> str: ...\n            ', runtime='def f4(a, *args, b, **kwargs): pass', error=None)
        yield Case(stub='\n            @overload\n            def f5(__a: int) -> int: ...\n            @overload\n            def f5(__b: str) -> str: ...\n            ', runtime='def f5(x, /): pass', error=None)
        yield Case(stub='\n            from typing import deprecated, final\n            class Foo:\n                @overload\n                @final\n                def f6(self, __a: int) -> int: ...\n                @overload\n                @deprecated("evil")\n                def f6(self, __b: str) -> str: ...\n            ', runtime='\n            class Foo:\n                def f6(self, x, /): pass\n            ', error=None)

    @collect_cases
    def test_property(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='\n            class Good:\n                @property\n                def read_only_attr(self) -> int: ...\n            ', runtime='\n            class Good:\n                @property\n                def read_only_attr(self): return 1\n            ', error=None)
        yield Case(stub='\n            class Bad:\n                @property\n                def f(self) -> int: ...\n            ', runtime='\n            class Bad:\n                def f(self) -> int: return 1\n            ', error='Bad.f')
        yield Case(stub='\n            class GoodReadOnly:\n                @property\n                def f(self) -> int: ...\n            ', runtime='\n            class GoodReadOnly:\n                f = 1\n            ', error=None)
        yield Case(stub='\n            class BadReadOnly:\n                @property\n                def f(self) -> str: ...\n            ', runtime='\n            class BadReadOnly:\n                f = 1\n            ', error='BadReadOnly.f')
        yield Case(stub='\n            class Y:\n                @property\n                def read_only_attr(self) -> int: ...\n                @read_only_attr.setter\n                def read_only_attr(self, val: int) -> None: ...\n            ', runtime='\n            class Y:\n                @property\n                def read_only_attr(self): return 5\n            ', error='Y.read_only_attr')
        yield Case(stub='\n            class Z:\n                @property\n                def read_write_attr(self) -> int: ...\n                @read_write_attr.setter\n                def read_write_attr(self, val: int) -> None: ...\n            ', runtime='\n            class Z:\n                @property\n                def read_write_attr(self): return self._val\n                @read_write_attr.setter\n                def read_write_attr(self, val): self._val = val\n            ', error=None)
        yield Case(stub='\n            class FineAndDandy:\n                @property\n                def attr(self) -> int: ...\n            ', runtime="\n            class _EvilDescriptor:\n                def __get__(self, instance, ownerclass=None):\n                    if instance is None:\n                        raise AttributeError('no')\n                    return 42\n                def __set__(self, instance, value):\n                    raise AttributeError('no')\n\n            class FineAndDandy:\n                attr = _EvilDescriptor()\n            ", error=None)

    @collect_cases
    def test_var(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='x1: int', runtime='x1 = 5', error=None)
        yield Case(stub='x2: str', runtime='x2 = 5', error='x2')
        yield Case('from typing import Tuple', '', None)
        yield Case(stub='\n            x3: Tuple[int, int]\n            ', runtime='x3 = (1, 3)', error=None)
        yield Case(stub='\n            x4: Tuple[int, int]\n            ', runtime='x4 = (1, 3, 5)', error='x4')
        yield Case(stub='x5: int', runtime='def x5(a, b): pass', error='x5')
        yield Case(stub='def foo(a: int, b: int) -> None: ...\nx6 = foo', runtime='def foo(a, b): pass\ndef x6(c, d): pass', error='x6')
        yield Case(stub='\n            class X:\n                f: int\n            ', runtime='\n            class X:\n                def __init__(self):\n                    self.f = "asdf"\n            ', error=None)
        yield Case(stub='\n            class Y:\n                read_only_attr: int\n            ', runtime='\n            class Y:\n                @property\n                def read_only_attr(self): return 5\n            ', error='Y.read_only_attr')
        yield Case(stub='\n            class Z:\n                read_write_attr: int\n            ', runtime='\n            class Z:\n                @property\n                def read_write_attr(self): return self._val\n                @read_write_attr.setter\n                def read_write_attr(self, val): self._val = val\n            ', error=None)

    @collect_cases
    def test_type_alias(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='\n            import collections.abc\n            import re\n            import typing\n            from typing import Callable, Dict, Generic, Iterable, List, Match, Tuple, TypeVar, Union\n            ', runtime='\n            import collections.abc\n            import re\n            from typing import Callable, Dict, Generic, Iterable, List, Match, Tuple, TypeVar, Union\n            ', error=None)
        yield Case(stub='\n            class X:\n                def f(self) -> None: ...\n            Y = X\n            ', runtime='\n            class X:\n                def f(self) -> None: ...\n            class Y: ...\n            ', error='Y.f')
        yield Case(stub='A = Tuple[int, str]', runtime='A = (int, str)', error='A')
        yield Case(stub='B = str', runtime='', error='B')
        yield Case(stub='_C = int', runtime='', error=None)
        yield Case(stub='\n            D = tuple[str, str]\n            E = Tuple[int, int, int]\n            F = Tuple[str, int]\n            ', runtime='\n            D = Tuple[str, str]\n            E = Tuple[int, int, int]\n            F = List[str]\n            ', error='F')
        yield Case(stub='\n            G = str | int\n            H = Union[str, bool]\n            I = str | int\n            ', runtime='\n            G = Union[str, int]\n            H = Union[str, bool]\n            I = str\n            ', error='I')
        yield Case(stub='\n            K = dict[str, str]\n            L = Dict[int, int]\n            KK = collections.abc.Iterable[str]\n            LL = typing.Iterable[str]\n            ', runtime='\n            K = Dict[str, str]\n            L = Dict[int, int]\n            KK = Iterable[str]\n            LL = Iterable[str]\n            ', error=None)
        yield Case(stub='\n            _T = TypeVar("_T")\n            class _Spam(Generic[_T]):\n                def foo(self) -> None: ...\n            IntFood = _Spam[int]\n            ', runtime='\n            _T = TypeVar("_T")\n            class _Bacon(Generic[_T]):\n                def foo(self, arg): pass\n            IntFood = _Bacon[int]\n            ', error='IntFood.foo')
        yield Case(stub='StrList = list[str]', runtime="StrList = ['foo', 'bar']", error='StrList')
        yield Case(stub='\n            N = typing.Callable[[str], bool]\n            O = collections.abc.Callable[[int], str]\n            P = typing.Callable[[str], bool]\n            ', runtime='\n            N = Callable[[str], bool]\n            O = Callable[[int], str]\n            P = int\n            ', error='P')
        yield Case(stub='\n            class Foo:\n                class Bar: ...\n            BarAlias = Foo.Bar\n            ', runtime='\n            class Foo:\n                class Bar: pass\n            BarAlias = Foo.Bar\n            ', error=None)
        yield Case(stub='\n            from io import StringIO\n            StringIOAlias = StringIO\n            ', runtime='\n            from _io import StringIO\n            StringIOAlias = StringIO\n            ', error=None)
        yield Case(stub='M = Match[str]', runtime='M = Match[str]', error=None)
        yield Case(stub='\n            class Baz:\n                def fizz(self) -> None: ...\n            BazAlias = Baz\n            ', runtime='\n            class Baz:\n                def fizz(self): pass\n            BazAlias = Baz\n            Baz.__name__ = Baz.__qualname__ = Baz.__module__ = "New"\n            ', error=None)
        yield Case(stub='\n            class FooBar:\n                __module__: None  # type: ignore\n                def fizz(self) -> None: ...\n            FooBarAlias = FooBar\n            ', runtime='\n            class FooBar:\n                def fizz(self): pass\n            FooBarAlias = FooBar\n            FooBar.__module__ = None\n            ', error=None)
        if sys.version_info >= (3, 10):
            yield Case(stub='\n                Q = Dict[str, str]\n                R = dict[int, int]\n                S = Tuple[int, int]\n                T = tuple[str, str]\n                U = int | str\n                V = Union[int, str]\n                W = typing.Callable[[str], bool]\n                Z = collections.abc.Callable[[str], bool]\n                QQ = typing.Iterable[str]\n                RR = collections.abc.Iterable[str]\n                MM = typing.Match[str]\n                MMM = re.Match[str]\n                ', runtime='\n                Q = dict[str, str]\n                R = dict[int, int]\n                S = tuple[int, int]\n                T = tuple[str, str]\n                U = int | str\n                V = int | str\n                W = collections.abc.Callable[[str], bool]\n                Z = collections.abc.Callable[[str], bool]\n                QQ = collections.abc.Iterable[str]\n                RR = collections.abc.Iterable[str]\n                MM = re.Match[str]\n                MMM = re.Match[str]\n                ', error=None)

    @collect_cases
    def test_enum(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='import enum', runtime='import enum', error=None)
        yield Case(stub='\n            class X(enum.Enum):\n                a: int\n                b: str\n                c: str\n            ', runtime='\n            class X(enum.Enum):\n                a = 1\n                b = "asdf"\n                c = 2\n            ', error='X.c')
        yield Case(stub='\n            class Flags1(enum.Flag):\n                a: int\n                b: int\n            def foo(x: Flags1 = ...) -> None: ...\n            ', runtime='\n            class Flags1(enum.Flag):\n                a = 1\n                b = 2\n            def foo(x=Flags1.a|Flags1.b): pass\n            ', error=None)
        yield Case(stub='\n            class Flags2(enum.Flag):\n                a: int\n                b: int\n            def bar(x: Flags2 | None = None) -> None: ...\n            ', runtime='\n            class Flags2(enum.Flag):\n                a = 1\n                b = 2\n            def bar(x=Flags2.a|Flags2.b): pass\n            ', error='bar')
        yield Case(stub='\n            class Flags3(enum.Flag):\n                a: int\n                b: int\n            def baz(x: Flags3 | None = ...) -> None: ...\n            ', runtime='\n            class Flags3(enum.Flag):\n                a = 1\n                b = 2\n            def baz(x=Flags3(0)): pass\n            ', error=None)
        yield Case(stub='\n            class Flags4(enum.Flag):\n                a: int\n                b: int\n            def spam(x: Flags4 | None = None) -> None: ...\n            ', runtime='\n            class Flags4(enum.Flag):\n                a = 1\n                b = 2\n            def spam(x=Flags4(0)): pass\n            ', error='spam')
        yield Case(stub='\n            from typing_extensions import Final, Literal\n            class BytesEnum(bytes, enum.Enum):\n                a: bytes\n            FOO: Literal[BytesEnum.a]\n            BAR: Final = BytesEnum.a\n            BAZ: BytesEnum\n            EGGS: bytes\n            ', runtime="\n            class BytesEnum(bytes, enum.Enum):\n                a = b'foo'\n            FOO = BytesEnum.a\n            BAR = BytesEnum.a\n            BAZ = BytesEnum.a\n            EGGS = BytesEnum.a\n            ", error=None)

    @collect_cases
    def test_decorator(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='\n            from typing import Any, Callable\n            def decorator(f: Callable[[], int]) -> Callable[..., Any]: ...\n            @decorator\n            def f() -> Any: ...\n            ', runtime='\n            def decorator(f): return f\n            @decorator\n            def f(): return 3\n            ', error=None)

    @collect_cases
    def test_all_at_runtime_not_stub(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='Z: int', runtime='\n            __all__ = []\n            Z = 5', error=None)

    @collect_cases
    def test_all_in_stub_not_at_runtime(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='__all__ = ()', runtime='', error='__all__')

    @collect_cases
    def test_all_in_stub_different_to_all_at_runtime(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub="\n            __all__ = ['foo']\n            foo: str\n            ", runtime="\n            __all__ = []\n            foo = 'foo'\n            ", error='__all__')

    @collect_cases
    def test_missing(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='x = 5', runtime='', error='x')
        yield Case(stub='def f(): ...', runtime='', error='f')
        yield Case(stub='class X: ...', runtime='', error='X')
        yield Case(stub='\n            from typing import overload\n            @overload\n            def h(x: int): ...\n            @overload\n            def h(x: str): ...\n            ', runtime='', error='h')
        yield Case(stub='', runtime='__all__ = []', error=None)
        yield Case(stub='', runtime="__all__ += ['y']\ny = 5", error='y')
        yield Case(stub='', runtime="__all__ += ['g']\ndef g(): pass", error='g')
        yield Case(stub='from mystery import A, B as B, C as D  # type: ignore', runtime='', error='B')
        yield Case(stub='class Y: ...', runtime="__all__ += ['Y']\nclass Y:\n  def __or__(self, other): return self|other", error='Y.__or__')
        yield Case(stub='class Z: ...', runtime="__all__ += ['Z']\nclass Z:\n  def __reduce__(self): return (Z,)", error=None)

    @collect_cases
    def test_missing_no_runtime_all(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='', runtime='import sys', error=None)
        yield Case(stub='', runtime='def g(): ...', error='g')
        yield Case(stub='', runtime='CONSTANT = 0', error='CONSTANT')
        yield Case(stub='', runtime="import re; constant = re.compile('foo')", error='constant')
        yield Case(stub='', runtime='from json.scanner import NUMBER_RE', error=None)
        yield Case(stub='', runtime='from string import ascii_letters', error=None)

    @collect_cases
    def test_non_public_1(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='__all__: list[str]', runtime='', error=f'{TEST_MODULE_NAME}.__all__')
        yield Case(stub='_f: int', runtime='def _f(): ...', error='_f')

    @collect_cases
    def test_non_public_2(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub="__all__: list[str] = ['f']", runtime="__all__ = ['f']", error=None)
        yield Case(stub='f: int', runtime='def f(): ...', error='f')
        yield Case(stub='g: int', runtime='def g(): ...', error='g')

    @collect_cases
    def test_dunders(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='class A:\n  def __init__(self, a: int, b: int) -> None: ...', runtime='class A:\n  def __init__(self, a, bx): pass', error='A.__init__')
        yield Case(stub='class B:\n  def __call__(self, c: int, d: int) -> None: ...', runtime='class B:\n  def __call__(self, c, dx): pass', error='B.__call__')
        yield Case(stub='class C:\n  def __init_subclass__(\n    cls, e: int = ..., **kwargs: int\n  ) -> None: ...\n', runtime='class C:\n  def __init_subclass__(cls, e=1, **kwargs): pass', error=None)
        if sys.version_info >= (3, 9):
            yield Case(stub='class D:\n  def __class_getitem__(cls, type: type) -> type: ...', runtime='class D:\n  def __class_getitem__(cls, type): ...', error=None)

    @collect_cases
    def test_not_subclassable(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='class CanBeSubclassed: ...', runtime='class CanBeSubclassed: ...', error=None)
        yield Case(stub='class CannotBeSubclassed:\n  def __init_subclass__(cls) -> None: ...', runtime='class CannotBeSubclassed:\n  def __init_subclass__(cls): raise TypeError', error='CannotBeSubclassed')

    @collect_cases
    def test_has_runtime_final_decorator(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='from typing_extensions import final', runtime='\n            import functools\n            from typing_extensions import final\n            ', error=None)
        yield Case(stub='\n            @final\n            class A: ...\n            ', runtime='\n            @final\n            class A: ...\n            ', error=None)
        yield Case(stub='\n            @final\n            class B: ...\n            ', runtime='\n            class B: ...\n            ', error=None)
        yield Case(stub='\n            class C: ...\n            ', runtime='\n            @final\n            class C: ...\n            ', error='C')
        yield Case(stub='\n            class D:\n                @final\n                def foo(self) -> None: ...\n                @final\n                @staticmethod\n                def bar() -> None: ...\n                @staticmethod\n                @final\n                def bar2() -> None: ...\n                @final\n                @classmethod\n                def baz(cls) -> None: ...\n                @classmethod\n                @final\n                def baz2(cls) -> None: ...\n                @property\n                @final\n                def eggs(self) -> int: ...\n                @final\n                @property\n                def eggs2(self) -> int: ...\n                @final\n                def ham(self, obj: int) -> int: ...\n            ', runtime='\n            class D:\n                @final\n                def foo(self): pass\n                @final\n                @staticmethod\n                def bar(): pass\n                @staticmethod\n                @final\n                def bar2(): pass\n                @final\n                @classmethod\n                def baz(cls): pass\n                @classmethod\n                @final\n                def baz2(cls): pass\n                @property\n                @final\n                def eggs(self): return 42\n                @final\n                @property\n                def eggs2(self): pass\n                @final\n                @functools.lru_cache()\n                def ham(self, obj): return obj * 2\n            ', error=None)
        yield Case(stub='\n            class E:\n                @final\n                def foo(self) -> None: ...\n                @final\n                @staticmethod\n                def bar() -> None: ...\n                @staticmethod\n                @final\n                def bar2() -> None: ...\n                @final\n                @classmethod\n                def baz(cls) -> None: ...\n                @classmethod\n                @final\n                def baz2(cls) -> None: ...\n                @property\n                @final\n                def eggs(self) -> int: ...\n                @final\n                @property\n                def eggs2(self) -> int: ...\n                @final\n                def ham(self, obj: int) -> int: ...\n            ', runtime='\n            class E:\n                def foo(self): pass\n                @staticmethod\n                def bar(): pass\n                @staticmethod\n                def bar2(): pass\n                @classmethod\n                def baz(cls): pass\n                @classmethod\n                def baz2(cls): pass\n                @property\n                def eggs(self): return 42\n                @property\n                def eggs2(self): return 42\n                @functools.lru_cache()\n                def ham(self, obj): return obj * 2\n            ', error=None)
        yield Case(stub='\n            class F:\n                def foo(self) -> None: ...\n            ', runtime='\n            class F:\n                @final\n                def foo(self): pass\n            ', error='F.foo')
        yield Case(stub='\n            class G:\n                @staticmethod\n                def foo() -> None: ...\n            ', runtime='\n            class G:\n                @final\n                @staticmethod\n                def foo(): pass\n            ', error='G.foo')
        yield Case(stub='\n            class H:\n                @staticmethod\n                def foo() -> None: ...\n            ', runtime='\n            class H:\n                @staticmethod\n                @final\n                def foo(): pass\n            ', error='H.foo')
        yield Case(stub='\n            class I:\n                @classmethod\n                def foo(cls) -> None: ...\n            ', runtime='\n            class I:\n                @final\n                @classmethod\n                def foo(cls): pass\n            ', error='I.foo')
        yield Case(stub='\n            class J:\n                @classmethod\n                def foo(cls) -> None: ...\n            ', runtime='\n            class J:\n                @classmethod\n                @final\n                def foo(cls): pass\n            ', error='J.foo')
        yield Case(stub='\n            class K:\n                @property\n                def foo(self) -> int: ...\n            ', runtime='\n            class K:\n                @property\n                @final\n                def foo(self): return 42\n            ', error='K.foo')
        yield Case(stub='\n            class L:\n                def foo(self, obj: int) -> int: ...\n            ', runtime='\n            class L:\n                @final\n                @functools.lru_cache()\n                def foo(self, obj): return obj * 2\n            ', error='L.foo')

    @collect_cases
    def test_name_mangling(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='\n            class X:\n                def __mangle_good(self, text: str) -> None: ...\n                def __mangle_bad(self, number: int) -> None: ...\n            ', runtime='\n            class X:\n                def __mangle_good(self, text): pass\n                def __mangle_bad(self, text): pass\n            ', error='X.__mangle_bad')
        yield Case(stub='\n            class Klass:\n                class __Mangled1:\n                    class __Mangled2:\n                        def __mangle_good(self, text: str) -> None: ...\n                        def __mangle_bad(self, number: int) -> None: ...\n            ', runtime='\n            class Klass:\n                class __Mangled1:\n                    class __Mangled2:\n                        def __mangle_good(self, text): pass\n                        def __mangle_bad(self, text): pass\n            ', error='Klass.__Mangled1.__Mangled2.__mangle_bad')
        yield Case(stub='\n            class __Dunder__:\n                def __mangle_good(self, text: str) -> None: ...\n                def __mangle_bad(self, number: int) -> None: ...\n            ', runtime='\n            class __Dunder__:\n                def __mangle_good(self, text): pass\n                def __mangle_bad(self, text): pass\n            ', error='__Dunder__.__mangle_bad')
        yield Case(stub='\n            class _Private:\n                def __mangle_good(self, text: str) -> None: ...\n                def __mangle_bad(self, number: int) -> None: ...\n            ', runtime='\n            class _Private:\n                def __mangle_good(self, text): pass\n                def __mangle_bad(self, text): pass\n            ', error='_Private.__mangle_bad')

    @collect_cases
    def test_mro(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='\n            class A:\n                def foo(self, x: int) -> None: ...\n            class B(A):\n                pass\n            class C(A):\n                pass\n            ', runtime='\n            class A:\n                def foo(self, x: int) -> None: ...\n            class B(A):\n                def foo(self, x: int) -> None: ...\n            class C(A):\n                def foo(self, y: int) -> None: ...\n            ', error='C.foo')
        yield Case(stub='\n            class X: ...\n            ', runtime='\n            class X:\n                def __init__(self, x): pass\n            ', error='X.__init__')

    @collect_cases
    def test_good_literal(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub="\n            from typing_extensions import Literal\n\n            import enum\n            class Color(enum.Enum):\n                RED: int\n\n            NUM: Literal[1]\n            CHAR: Literal['a']\n            FLAG: Literal[True]\n            NON: Literal[None]\n            BYT1: Literal[b'abc']\n            BYT2: Literal[b'\\x90']\n            ENUM: Literal[Color.RED]\n            ", runtime='\n            import enum\n            class Color(enum.Enum):\n                RED = 3\n\n            NUM = 1\n            CHAR = \'a\'\n            NON = None\n            FLAG = True\n            BYT1 = b"abc"\n            BYT2 = b\'\\x90\'\n            ENUM = Color.RED\n            ', error=None)

    @collect_cases
    def test_bad_literal(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case('from typing_extensions import Literal', '', None)
        yield Case(stub='INT_FLOAT_MISMATCH: Literal[1]', runtime='INT_FLOAT_MISMATCH = 1.0', error='INT_FLOAT_MISMATCH')
        yield Case(stub='WRONG_INT: Literal[1]', runtime='WRONG_INT = 2', error='WRONG_INT')
        yield Case(stub="WRONG_STR: Literal['a']", runtime="WRONG_STR = 'b'", error='WRONG_STR')
        yield Case(stub="BYTES_STR_MISMATCH: Literal[b'value']", runtime="BYTES_STR_MISMATCH = 'value'", error='BYTES_STR_MISMATCH')
        yield Case(stub="STR_BYTES_MISMATCH: Literal['value']", runtime="STR_BYTES_MISMATCH = b'value'", error='STR_BYTES_MISMATCH')
        yield Case(stub="WRONG_BYTES: Literal[b'abc']", runtime="WRONG_BYTES = b'xyz'", error='WRONG_BYTES')
        yield Case(stub='WRONG_BOOL_1: Literal[True]', runtime='WRONG_BOOL_1 = False', error='WRONG_BOOL_1')
        yield Case(stub='WRONG_BOOL_2: Literal[False]', runtime='WRONG_BOOL_2 = True', error='WRONG_BOOL_2')

    @collect_cases
    def test_special_subtype(self) -> Iterator[Case]:
        if False:
            i = 10
            return i + 15
        yield Case(stub='\n            b1: bool\n            b2: bool\n            b3: bool\n            ', runtime='\n            b1 = 0\n            b2 = 1\n            b3 = 2\n            ', error='b3')
        yield Case(stub='\n            from typing_extensions import TypedDict\n\n            class _Options(TypedDict):\n                a: str\n                b: int\n\n            opt1: _Options\n            opt2: _Options\n            opt3: _Options\n            ', runtime='\n            opt1 = {"a": "3.", "b": 14}\n            opt2 = {"some": "stuff"}  # false negative\n            opt3 = 0\n            ', error='opt3')

    @collect_cases
    def test_runtime_typing_objects(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='from typing_extensions import Protocol, TypedDict', runtime='from typing_extensions import Protocol, TypedDict', error=None)
        yield Case(stub='\n            class X(Protocol):\n                bar: int\n                def foo(self, x: int, y: bytes = ...) -> str: ...\n            ', runtime='\n            class X(Protocol):\n                bar: int\n                def foo(self, x: int, y: bytes = ...) -> str: ...\n            ', error=None)
        yield Case(stub='\n            class Y(TypedDict):\n                a: int\n            ', runtime='\n            class Y(TypedDict):\n                a: int\n            ', error=None)

    @collect_cases
    def test_named_tuple(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='from typing import NamedTuple', runtime='from typing import NamedTuple', error=None)
        yield Case(stub='\n            class X1(NamedTuple):\n                bar: int\n                foo: str = ...\n            ', runtime="\n            class X1(NamedTuple):\n                bar: int\n                foo: str = 'a'\n            ", error=None)
        yield Case(stub='\n            class X2(NamedTuple):\n                bar: int\n                foo: str\n            ', runtime="\n            class X2(NamedTuple):\n                bar: int\n                foo: str = 'a'\n            ", error='X2.__new__')

    @collect_cases
    def test_named_tuple_typing_and_collections(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='from typing import NamedTuple', runtime='from collections import namedtuple', error=None)
        yield Case(stub='\n            class X1(NamedTuple):\n                bar: int\n                foo: str = ...\n            ', runtime="\n            X1 = namedtuple('X1', ['bar', 'foo'], defaults=['a'])\n            ", error=None)
        yield Case(stub='\n            class X2(NamedTuple):\n                bar: int\n                foo: str\n            ', runtime="\n            X2 = namedtuple('X1', ['bar', 'foo'], defaults=['a'])\n            ", error='X2.__new__')

    @collect_cases
    def test_type_var(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='from typing import TypeVar', runtime='from typing import TypeVar', error=None)
        yield Case(stub="A = TypeVar('A')", runtime="A = TypeVar('A')", error=None)
        yield Case(stub="B = TypeVar('B')", runtime='B = 5', error='B')
        if sys.version_info >= (3, 10):
            yield Case(stub='from typing import ParamSpec', runtime='from typing import ParamSpec', error=None)
            yield Case(stub="C = ParamSpec('C')", runtime="C = ParamSpec('C')", error=None)

    @collect_cases
    def test_metaclass_match(self) -> Iterator[Case]:
        if False:
            for i in range(10):
                print('nop')
        yield Case(stub='class Meta(type): ...', runtime='class Meta(type): ...', error=None)
        yield Case(stub='class A0: ...', runtime='class A0: ...', error=None)
        yield Case(stub='class A1(metaclass=Meta): ...', runtime='class A1(metaclass=Meta): ...', error=None)
        yield Case(stub='class A2: ...', runtime='class A2(metaclass=Meta): ...', error='A2')
        yield Case(stub='class A3(metaclass=Meta): ...', runtime='class A3: ...', error='A3')
        yield Case(stub='class T1(metaclass=type): ...', runtime='class T1(metaclass=type): ...', error=None)
        yield Case(stub='class T2: ...', runtime='class T2(metaclass=type): ...', error=None)
        yield Case(stub='class T3(metaclass=type): ...', runtime='class T3: ...', error=None)
        yield Case(stub='class _P1(type): ...', runtime='class _P1(type): ...', error=None)
        yield Case(stub='class P2: ...', runtime='class P2(metaclass=_P1): ...', error='P2')
        yield Case(stub='\n            class I1(metaclass=Meta): ...\n            class S1(I1): ...\n            ', runtime='\n            class I1(metaclass=Meta): ...\n            class S1(I1): ...\n            ', error=None)
        yield Case(stub='\n            class I2(metaclass=Meta): ...\n            class S2: ...  # missing inheritance\n            ', runtime='\n            class I2(metaclass=Meta): ...\n            class S2(I2): ...\n            ', error='S2')

    @collect_cases
    def test_metaclass_abcmeta(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='from abc import ABCMeta', runtime='from abc import ABCMeta', error=None)
        yield Case(stub='class A1(metaclass=ABCMeta): ...', runtime='class A1(metaclass=ABCMeta): ...', error=None)
        yield Case(stub='class A2: ...', runtime='class A2(metaclass=ABCMeta): ...', error='A2')
        yield Case(stub='class A3(metaclass=ABCMeta): ...', runtime='class A3: ...', error=None)

    @collect_cases
    def test_abstract_methods(self) -> Iterator[Case]:
        if False:
            print('Hello World!')
        yield Case(stub='\n            from abc import abstractmethod\n            from typing import overload\n            ', runtime='from abc import abstractmethod', error=None)
        yield Case(stub='\n            class A1:\n                def some(self) -> None: ...\n            ', runtime='\n            class A1:\n                @abstractmethod\n                def some(self) -> None: ...\n            ', error='A1.some')
        yield Case(stub='\n            class A2:\n                @abstractmethod\n                def some(self) -> None: ...\n            ', runtime='\n            class A2:\n                @abstractmethod\n                def some(self) -> None: ...\n            ', error=None)
        yield Case(stub='\n            class A3:\n                @overload\n                def some(self, other: int) -> str: ...\n                @overload\n                def some(self, other: str) -> int: ...\n            ', runtime='\n            class A3:\n                @abstractmethod\n                def some(self, other) -> None: ...\n            ', error='A3.some')
        yield Case(stub='\n            class A4:\n                @overload\n                @abstractmethod\n                def some(self, other: int) -> str: ...\n                @overload\n                @abstractmethod\n                def some(self, other: str) -> int: ...\n            ', runtime='\n            class A4:\n                @abstractmethod\n                def some(self, other) -> None: ...\n            ', error=None)
        yield Case(stub='\n            class A5:\n                @abstractmethod\n                @overload\n                def some(self, other: int) -> str: ...\n                @abstractmethod\n                @overload\n                def some(self, other: str) -> int: ...\n            ', runtime='\n            class A5:\n                @abstractmethod\n                def some(self, other) -> None: ...\n            ', error=None)
        yield Case(stub='\n            class A6:\n                @abstractmethod\n                def some(self) -> None: ...\n            ', runtime='\n            class A6:\n                def some(self) -> None: ...\n            ', error=None)

    @collect_cases
    def test_abstract_properties(self) -> Iterator[Case]:
        if False:
            return 10
        yield Case(stub='from abc import abstractmethod', runtime='from abc import abstractmethod', error=None)
        yield Case(stub='\n            class AP1:\n                @property\n                def some(self) -> int: ...\n            ', runtime='\n            class AP1:\n                @property\n                @abstractmethod\n                def some(self) -> int: ...\n            ', error='AP1.some')
        yield Case(stub='\n            class AP1_2:\n                def some(self) -> int: ...  # missing `@property` decorator\n            ', runtime='\n            class AP1_2:\n                @property\n                @abstractmethod\n                def some(self) -> int: ...\n            ', error='AP1_2.some')
        yield Case(stub='\n            class AP2:\n                @property\n                @abstractmethod\n                def some(self) -> int: ...\n            ', runtime='\n            class AP2:\n                @property\n                @abstractmethod\n                def some(self) -> int: ...\n            ', error=None)
        yield Case(stub='\n            class AP3:\n                @property\n                @abstractmethod\n                def some(self) -> int: ...\n            ', runtime='\n            class AP3:\n                @property\n                def some(self) -> int: ...\n            ', error=None)

    @collect_cases
    def test_type_check_only(self) -> Iterator[Case]:
        if False:
            while True:
                i = 10
        yield Case(stub='from typing import type_check_only, overload', runtime='from typing import overload', error=None)
        yield Case(stub='\n            @type_check_only\n            class A1: ...\n            ', runtime='', error=None)
        yield Case(stub='\n            @type_check_only\n            class A2: ...\n            ', runtime='class A2: ...', error='A2')
        yield Case(stub='from typing_extensions import NamedTuple, TypedDict', runtime='from typing_extensions import NamedTuple, TypedDict', error=None)
        yield Case(stub='\n            @type_check_only\n            class NT1(NamedTuple): ...\n            ', runtime='class NT1(NamedTuple): ...', error='NT1')
        yield Case(stub='\n            @type_check_only\n            class TD1(TypedDict): ...\n            ', runtime='class TD1(TypedDict): ...', error='TD1')
        yield Case(stub='\n            @type_check_only\n            def func1() -> None: ...\n            ', runtime='', error=None)
        yield Case(stub='\n            @type_check_only\n            def func2() -> None: ...\n            ', runtime='def func2() -> None: ...', error='func2')

def remove_color_code(s: str) -> str:
    if False:
        while True:
            i = 10
    return re.sub('\\x1b.*?m', '', s)

class StubtestMiscUnit(unittest.TestCase):

    def test_output(self) -> None:
        if False:
            return 10
        output = run_stubtest(stub='def bad(number: int, text: str) -> None: ...', runtime='def bad(num, text): pass', options=[])
        expected = f'error: {TEST_MODULE_NAME}.bad is inconsistent, stub argument "number" differs from runtime argument "num"\nStub: in file {TEST_MODULE_NAME}.pyi:1\ndef (number: builtins.int, text: builtins.str)\nRuntime: in file {TEST_MODULE_NAME}.py:1\ndef (num, text)\n\nFound 1 error (checked 1 module)\n'
        assert output == expected
        output = run_stubtest(stub='def bad(number: int, text: str) -> None: ...', runtime='def bad(num, text): pass', options=['--concise'])
        expected = '{}.bad is inconsistent, stub argument "number" differs from runtime argument "num"\n'.format(TEST_MODULE_NAME)
        assert output == expected

    def test_ignore_flags(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        output = run_stubtest(stub='', runtime="__all__ = ['f']\ndef f(): pass", options=['--ignore-missing-stub'])
        assert output == 'Success: no issues found in 1 module\n'
        output = run_stubtest(stub='', runtime='def f(): pass', options=['--ignore-missing-stub'])
        assert output == 'Success: no issues found in 1 module\n'
        output = run_stubtest(stub='def f(__a): ...', runtime='def f(a): pass', options=['--ignore-positional-only'])
        assert output == 'Success: no issues found in 1 module\n'

    def test_allowlist(self) -> None:
        if False:
            return 10
        allowlist = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        try:
            with allowlist:
                allowlist.write(f'{TEST_MODULE_NAME}.bad  # comment\n# comment')
            output = run_stubtest(stub='def bad(number: int, text: str) -> None: ...', runtime='def bad(asdf, text): pass', options=['--allowlist', allowlist.name])
            assert output == 'Success: no issues found in 1 module\n'
            output = run_stubtest(stub='', runtime='', options=['--allowlist', allowlist.name])
            assert output == f'note: unused allowlist entry {TEST_MODULE_NAME}.bad\nFound 1 error (checked 1 module)\n'
            output = run_stubtest(stub='', runtime='', options=['--allowlist', allowlist.name, '--ignore-unused-allowlist'])
            assert output == 'Success: no issues found in 1 module\n'
            with open(allowlist.name, mode='w+') as f:
                f.write(f'{TEST_MODULE_NAME}.b.*\n')
                f.write('(unused_missing)?\n')
                f.write('unused.*\n')
            output = run_stubtest(stub=textwrap.dedent('\n                    def good() -> None: ...\n                    def bad(number: int) -> None: ...\n                    def also_bad(number: int) -> None: ...\n                    '.lstrip('\n')), runtime=textwrap.dedent('\n                    def good(): pass\n                    def bad(asdf): pass\n                    def also_bad(asdf): pass\n                    '.lstrip('\n')), options=['--allowlist', allowlist.name, '--generate-allowlist'])
            assert output == f'note: unused allowlist entry unused.*\n{TEST_MODULE_NAME}.also_bad\n'
        finally:
            os.unlink(allowlist.name)

    def test_mypy_build(self) -> None:
        if False:
            while True:
                i = 10
        output = run_stubtest(stub='+', runtime='', options=[])
        assert output == 'error: not checking stubs due to failed mypy compile:\n{}.pyi:1: error: invalid syntax  [syntax]\n'.format(TEST_MODULE_NAME)
        output = run_stubtest(stub='def f(): ...\ndef f(): ...', runtime='', options=[])
        assert output == 'error: not checking stubs due to mypy build errors:\n{}.pyi:2: error: Name "f" already defined on line 1  [no-redef]\n'.format(TEST_MODULE_NAME)

    def test_missing_stubs(self) -> None:
        if False:
            i = 10
            return i + 15
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            test_stubs(parse_options(['not_a_module']))
        assert remove_color_code(output.getvalue()) == 'error: not_a_module failed to find stubs\nStub:\nMISSING\nRuntime:\nN/A\n\nFound 1 error (checked 1 module)\n'

    def test_only_py(self) -> None:
        if False:
            while True:
                i = 10
        with use_tmp_dir(TEST_MODULE_NAME):
            with open(f'{TEST_MODULE_NAME}.py', 'w') as f:
                f.write('a = 1')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                test_stubs(parse_options([TEST_MODULE_NAME]))
            output_str = remove_color_code(output.getvalue())
            assert output_str == 'Success: no issues found in 1 module\n'

    def test_get_typeshed_stdlib_modules(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        stdlib = mypy.stubtest.get_typeshed_stdlib_modules(None, (3, 7))
        assert 'builtins' in stdlib
        assert 'os' in stdlib
        assert 'os.path' in stdlib
        assert 'asyncio' in stdlib
        assert 'graphlib' not in stdlib
        assert 'formatter' in stdlib
        assert 'contextvars' in stdlib
        assert 'importlib.metadata' not in stdlib
        stdlib = mypy.stubtest.get_typeshed_stdlib_modules(None, (3, 10))
        assert 'graphlib' in stdlib
        assert 'formatter' not in stdlib
        assert 'importlib.metadata' in stdlib

    def test_signature(self) -> None:
        if False:
            while True:
                i = 10

        def f(a: int, b: int, *, c: int, d: int=0, **kwargs: Any) -> None:
            if False:
                return 10
            pass
        assert str(mypy.stubtest.Signature.from_inspect_signature(inspect.signature(f))) == 'def (a, b, *, c, d = ..., **kwargs)'

    def test_config_file(self) -> None:
        if False:
            return 10
        runtime = 'temp = 5\n'
        stub = 'from decimal import Decimal\ntemp: Decimal\n'
        config_file = f'[mypy]\nplugins={root_dir}/test-data/unit/plugins/decimal_to_int.py\n'
        output = run_stubtest(stub=stub, runtime=runtime, options=[])
        assert output == f'error: {TEST_MODULE_NAME}.temp variable differs from runtime type Literal[5]\nStub: in file {TEST_MODULE_NAME}.pyi:2\n_decimal.Decimal\nRuntime:\n5\n\nFound 1 error (checked 1 module)\n'
        output = run_stubtest(stub=stub, runtime=runtime, options=[], config_file=config_file)
        assert output == 'Success: no issues found in 1 module\n'

    def test_no_modules(self) -> None:
        if False:
            i = 10
            return i + 15
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            test_stubs(parse_options([]))
        assert remove_color_code(output.getvalue()) == 'error: no modules to check\n'

    def test_module_and_typeshed(self) -> None:
        if False:
            i = 10
            return i + 15
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            test_stubs(parse_options(['--check-typeshed', 'some_module']))
        assert remove_color_code(output.getvalue()) == 'error: cannot pass both --check-typeshed and a list of modules\n'