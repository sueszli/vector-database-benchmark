from __future__ import annotations
import gc
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path
from traceback import extract_tb, print_exception
from typing import TYPE_CHECKING, Callable, NoReturn
import pytest
from ... import TrioDeprecationWarning
from ..._core import open_nursery
from .._multierror import MultiError, NonBaseMultiError, concat_tb
from .tutil import slow
if TYPE_CHECKING:
    from types import TracebackType
if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

class NotHashableException(Exception):
    code: int | None = None

    def __init__(self, code: int) -> None:
        if False:
            return 10
        super().__init__()
        self.code = code

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if not isinstance(other, NotHashableException):
            return False
        return self.code == other.code

async def raise_nothashable(code: int) -> NoReturn:
    raise NotHashableException(code)

def raiser1() -> NoReturn:
    if False:
        i = 10
        return i + 15
    raiser1_2()

def raiser1_2() -> NoReturn:
    if False:
        return 10
    raiser1_3()

def raiser1_3() -> NoReturn:
    if False:
        i = 10
        return i + 15
    raise ValueError('raiser1_string')

def raiser2() -> NoReturn:
    if False:
        while True:
            i = 10
    raiser2_2()

def raiser2_2() -> NoReturn:
    if False:
        return 10
    raise KeyError('raiser2_string')

def raiser3() -> NoReturn:
    if False:
        while True:
            i = 10
    raise NameError

def get_exc(raiser: Callable[[], NoReturn]) -> BaseException:
    if False:
        return 10
    try:
        raiser()
    except Exception as exc:
        return exc
    raise AssertionError('raiser should always raise')

def get_tb(raiser: Callable[[], NoReturn]) -> TracebackType | None:
    if False:
        for i in range(10):
            print('nop')
    return get_exc(raiser).__traceback__

def test_concat_tb() -> None:
    if False:
        while True:
            i = 10
    tb1 = get_tb(raiser1)
    tb2 = get_tb(raiser2)
    entries1 = extract_tb(tb1)
    entries2 = extract_tb(tb2)
    tb12 = concat_tb(tb1, tb2)
    assert extract_tb(tb12) == entries1 + entries2
    tb21 = concat_tb(tb2, tb1)
    assert extract_tb(tb21) == entries2 + entries1
    assert extract_tb(concat_tb(None, tb1)) == entries1
    assert extract_tb(concat_tb(tb1, None)) == entries1
    assert concat_tb(None, None) is None
    assert extract_tb(get_tb(raiser1)) == entries1
    assert extract_tb(get_tb(raiser2)) == entries2

def test_MultiError() -> None:
    if False:
        i = 10
        return i + 15
    exc1 = get_exc(raiser1)
    exc2 = get_exc(raiser2)
    assert MultiError([exc1]) is exc1
    m = MultiError([exc1, exc2])
    assert m.exceptions == (exc1, exc2)
    assert 'ValueError' in str(m)
    assert 'ValueError' in repr(m)
    with pytest.raises(TypeError):
        MultiError(object())
    with pytest.raises(TypeError):
        MultiError([KeyError(), ValueError])

def test_MultiErrorOfSingleMultiError() -> None:
    if False:
        for i in range(10):
            print('nop')
    exceptions = (KeyError(), ValueError())
    a = MultiError(exceptions)
    b = MultiError([a])
    assert b == a
    assert b.exceptions == exceptions

async def test_MultiErrorNotHashable() -> None:
    exc1 = NotHashableException(42)
    exc2 = NotHashableException(4242)
    exc3 = ValueError()
    assert exc1 != exc2
    assert exc1 != exc3
    with pytest.raises(MultiError):
        async with open_nursery() as nursery:
            nursery.start_soon(raise_nothashable, 42)
            nursery.start_soon(raise_nothashable, 4242)

def test_MultiError_filter_NotHashable() -> None:
    if False:
        i = 10
        return i + 15
    excs = MultiError([NotHashableException(42), ValueError()])

    def handle_ValueError(exc: BaseException) -> BaseException | None:
        if False:
            return 10
        if isinstance(exc, ValueError):
            return None
        else:
            return exc
    with pytest.warns(TrioDeprecationWarning):
        filtered_excs = MultiError.filter(handle_ValueError, excs)
    assert isinstance(filtered_excs, NotHashableException)

def make_tree() -> MultiError:
    if False:
        for i in range(10):
            print('nop')
    exc1 = get_exc(raiser1)
    exc2 = get_exc(raiser2)
    exc3 = get_exc(raiser3)
    try:
        raise MultiError([exc1, exc2])
    except BaseException as m12:
        return MultiError([m12, exc3])

def assert_tree_eq(m1: BaseException | MultiError | None, m2: BaseException | MultiError | None) -> None:
    if False:
        while True:
            i = 10
    if m1 is None or m2 is None:
        assert m1 is m2
        return
    assert type(m1) is type(m2)
    assert extract_tb(m1.__traceback__) == extract_tb(m2.__traceback__)
    assert_tree_eq(m1.__cause__, m2.__cause__)
    assert_tree_eq(m1.__context__, m2.__context__)
    if isinstance(m1, MultiError):
        assert isinstance(m2, MultiError)
        assert len(m1.exceptions) == len(m2.exceptions)
        for (e1, e2) in zip(m1.exceptions, m2.exceptions):
            assert_tree_eq(e1, e2)

def test_MultiError_filter() -> None:
    if False:
        print('Hello World!')

    def null_handler(exc: BaseException) -> BaseException:
        if False:
            for i in range(10):
                print('nop')
        return exc
    m = make_tree()
    assert_tree_eq(m, m)
    with pytest.warns(TrioDeprecationWarning):
        assert MultiError.filter(null_handler, m) is m
    assert_tree_eq(m, make_tree())
    m = make_tree()
    try:
        raise ValueError
    except ValueError:
        with pytest.warns(TrioDeprecationWarning):
            assert MultiError.filter(null_handler, m) is m
    assert_tree_eq(m, make_tree())

    def simple_filter(exc: BaseException) -> BaseException | None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(exc, ValueError):
            return None
        if isinstance(exc, KeyError):
            return RuntimeError()
        return exc
    with pytest.warns(TrioDeprecationWarning):
        new_m = MultiError.filter(simple_filter, make_tree())
    assert isinstance(new_m, MultiError)
    assert len(new_m.exceptions) == 2
    assert isinstance(new_m.exceptions[0], RuntimeError)
    assert isinstance(new_m.exceptions[1], NameError)
    assert isinstance(new_m.exceptions[0].__context__, KeyError)
    orig = make_tree()
    assert isinstance(orig.exceptions[0], MultiError)
    assert isinstance(orig.exceptions[0].exceptions[1], KeyError)
    orig_extracted = extract_tb(orig.__traceback__) + extract_tb(orig.exceptions[0].__traceback__) + extract_tb(orig.exceptions[0].exceptions[1].__traceback__)

    def p(exc: BaseException) -> None:
        if False:
            print('Hello World!')
        print_exception(type(exc), exc, exc.__traceback__)
    p(orig)
    p(orig.exceptions[0])
    p(orig.exceptions[0].exceptions[1])
    p(new_m.exceptions[0].__context__)
    assert new_m.__traceback__ is None
    new_extracted = extract_tb(new_m.exceptions[0].__context__.__traceback__)
    assert orig_extracted == new_extracted

    def filter_NameError(exc: BaseException) -> BaseException | None:
        if False:
            while True:
                i = 10
        if isinstance(exc, NameError):
            return None
        return exc
    m = make_tree()
    with pytest.warns(TrioDeprecationWarning):
        new_m = MultiError.filter(filter_NameError, m)
    assert new_m is m.exceptions[0]

    def filter_all(exc: BaseException) -> None:
        if False:
            for i in range(10):
                print('nop')
        return None
    with pytest.warns(TrioDeprecationWarning):
        assert MultiError.filter(filter_all, make_tree()) is None

def test_MultiError_catch() -> None:
    if False:
        i = 10
        return i + 15

    def noop(_: object) -> None:
        if False:
            print('Hello World!')
        pass
    with pytest.warns(TrioDeprecationWarning), MultiError.catch(noop):
        pass
    m = make_tree()
    with pytest.raises(MultiError) as excinfo:
        with pytest.warns(TrioDeprecationWarning), MultiError.catch(lambda exc: exc):
            raise m
    assert excinfo.value is m
    assert m.__traceback__ is not None
    assert m.__traceback__.tb_frame.f_code.co_name == 'test_MultiError_catch'
    assert m.__traceback__.tb_next is None
    m.__traceback__ = None
    assert_tree_eq(m, make_tree())
    with pytest.warns(TrioDeprecationWarning), MultiError.catch(lambda _: None):
        raise make_tree()

    def simple_filter(exc):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(exc, ValueError):
            return None
        if isinstance(exc, KeyError):
            return RuntimeError()
        return exc
    with pytest.raises(MultiError) as excinfo:
        with pytest.warns(TrioDeprecationWarning), MultiError.catch(simple_filter):
            raise make_tree()
    new_m = excinfo.value
    assert isinstance(new_m, MultiError)
    assert len(new_m.exceptions) == 2
    assert isinstance(new_m.exceptions[0], RuntimeError)
    assert isinstance(new_m.exceptions[1], NameError)
    assert not new_m.__suppress_context__
    assert new_m.__context__ is None
    v = ValueError()
    v.__cause__ = KeyError()
    with pytest.raises(ValueError) as excinfo:
        with pytest.warns(TrioDeprecationWarning), MultiError.catch(lambda exc: exc):
            raise v
    assert isinstance(excinfo.value.__cause__, KeyError)
    v = ValueError()
    context = KeyError()
    v.__context__ = context
    with pytest.raises(ValueError) as excinfo:
        with pytest.warns(TrioDeprecationWarning), MultiError.catch(lambda exc: exc):
            raise v
    assert excinfo.value.__context__ is context
    assert not excinfo.value.__suppress_context__
    for suppress_context in [True, False]:
        v = ValueError()
        context = KeyError()
        v.__context__ = context
        v.__suppress_context__ = suppress_context
        distractor = RuntimeError()
        with pytest.raises(ValueError) as excinfo:

            def catch_RuntimeError(exc):
                if False:
                    return 10
                if isinstance(exc, RuntimeError):
                    return None
                else:
                    return exc
            with pytest.warns(TrioDeprecationWarning):
                with MultiError.catch(catch_RuntimeError):
                    raise MultiError([v, distractor])
        assert excinfo.value.__context__ is context
        assert excinfo.value.__suppress_context__ == suppress_context

@pytest.mark.skipif(sys.implementation.name != 'cpython', reason='Only makes sense with refcounting GC')
def test_MultiError_catch_doesnt_create_cyclic_garbage() -> None:
    if False:
        return 10
    gc.collect()
    old_flags = gc.get_debug()

    def make_multi() -> NoReturn:
        if False:
            while True:
                i = 10
        raise MultiError([get_exc(raiser1), get_exc(raiser2)])

    def simple_filter(exc: BaseException) -> Exception | RuntimeError:
        if False:
            return 10
        if isinstance(exc, ValueError):
            return Exception()
        if isinstance(exc, KeyError):
            return RuntimeError()
        raise AssertionError('only ValueError and KeyError should exist')
    try:
        gc.set_debug(gc.DEBUG_SAVEALL)
        with pytest.raises(MultiError):
            with pytest.warns(TrioDeprecationWarning), MultiError.catch(simple_filter):
                raise make_multi()
        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()

def assert_match_in_seq(pattern_list: list[str], string: str) -> None:
    if False:
        i = 10
        return i + 15
    offset = 0
    print('looking for pattern matches...')
    for pattern in pattern_list:
        print('checking pattern:', pattern)
        reobj = re.compile(pattern)
        match = reobj.search(string, offset)
        assert match is not None
        offset = match.end()

def test_assert_match_in_seq() -> None:
    if False:
        while True:
            i = 10
    assert_match_in_seq(['a', 'b'], 'xx a xx b xx')
    assert_match_in_seq(['b', 'a'], 'xx b xx a xx')
    with pytest.raises(AssertionError):
        assert_match_in_seq(['a', 'b'], 'xx b xx a xx')

def test_base_multierror() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that MultiError() with at least one base exception will return a MultiError\n    object.\n    '
    exc = MultiError([ZeroDivisionError(), KeyboardInterrupt()])
    assert type(exc) is MultiError

def test_non_base_multierror() -> None:
    if False:
        while True:
            i = 10
    '\n    Test that MultiError() without base exceptions will return a NonBaseMultiError\n    object.\n    '
    exc = MultiError([ZeroDivisionError(), ValueError()])
    assert type(exc) is NonBaseMultiError
    assert isinstance(exc, ExceptionGroup)

def run_script(name: str) -> subprocess.CompletedProcess[bytes]:
    if False:
        while True:
            i = 10
    import trio
    trio_path = Path(trio.__file__).parent.parent
    script_path = Path(__file__).parent / 'test_multierror_scripts' / name
    env = dict(os.environ)
    print('parent PYTHONPATH:', env.get('PYTHONPATH'))
    pp = []
    if 'PYTHONPATH' in env:
        pp = env['PYTHONPATH'].split(os.pathsep)
    pp.insert(0, str(trio_path))
    pp.insert(0, str(script_path.parent))
    env['PYTHONPATH'] = os.pathsep.join(pp)
    print('subprocess PYTHONPATH:', env.get('PYTHONPATH'))
    cmd = [sys.executable, '-u', str(script_path)]
    print('running:', cmd)
    completed = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print('process output:')
    print(completed.stdout.decode('utf-8'))
    return completed

@slow
@pytest.mark.skipif(not Path('/usr/lib/python3/dist-packages/apport_python_hook.py').exists(), reason='need Ubuntu with python3-apport installed')
def test_apport_excepthook_monkeypatch_interaction() -> None:
    if False:
        return 10
    completed = run_script('apport_excepthook.py')
    stdout = completed.stdout.decode('utf-8')
    assert 'custom sys.excepthook' not in stdout
    assert_match_in_seq(['--- 1 ---', 'KeyError', '--- 2 ---', 'ValueError'], stdout)

@pytest.mark.parametrize('protocol', range(0, pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_multierror(protocol: int) -> None:
    if False:
        print('Hello World!')
    import trio
    my_except = ZeroDivisionError()
    try:
        1 / 0
    except ZeroDivisionError as exc:
        my_except = exc
    for (cls, errors) in ((ZeroDivisionError, [my_except]), (NonBaseMultiError, [my_except, ValueError()]), (MultiError, [BaseException(), my_except])):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', TrioDeprecationWarning)
            me = trio.MultiError(errors)
            dump = pickle.dumps(me, protocol=protocol)
            load = pickle.loads(dump)
        assert repr(me) == repr(load)
        assert me.__class__ == load.__class__ == cls
        assert me.__dict__.keys() == load.__dict__.keys()
        for (me_val, load_val) in zip(me.__dict__.values(), load.__dict__.values()):
            assert repr(me_val) == repr(load_val)