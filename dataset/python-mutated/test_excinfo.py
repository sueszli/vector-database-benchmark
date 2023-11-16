from __future__ import annotations
import importlib
import io
import operator
import queue
import re
import sys
import textwrap
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING
import _pytest._code
import pytest
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import FormattedExcinfo
from _pytest._io import TerminalWriter
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pytester import LineMatcher
from _pytest.pytester import Pytester
if TYPE_CHECKING:
    from _pytest._code.code import _TracebackStyle
if sys.version_info[:2] < (3, 11):
    from exceptiongroup import ExceptionGroup

@pytest.fixture
def limited_recursion_depth():
    if False:
        for i in range(10):
            print('nop')
    before = sys.getrecursionlimit()
    sys.setrecursionlimit(150)
    yield
    sys.setrecursionlimit(before)

def test_excinfo_simple() -> None:
    if False:
        for i in range(10):
            print('nop')
    try:
        raise ValueError
    except ValueError:
        info = _pytest._code.ExceptionInfo.from_current()
    assert info.type == ValueError

def test_excinfo_from_exc_info_simple() -> None:
    if False:
        for i in range(10):
            print('nop')
    try:
        raise ValueError
    except ValueError as e:
        assert e.__traceback__ is not None
        info = _pytest._code.ExceptionInfo.from_exc_info((type(e), e, e.__traceback__))
    assert info.type == ValueError

def test_excinfo_from_exception_simple() -> None:
    if False:
        return 10
    try:
        raise ValueError
    except ValueError as e:
        assert e.__traceback__ is not None
        info = _pytest._code.ExceptionInfo.from_exception(e)
    assert info.type == ValueError

def test_excinfo_from_exception_missing_traceback_assertion() -> None:
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError, match='must have.*__traceback__'):
        _pytest._code.ExceptionInfo.from_exception(ValueError())

def test_excinfo_getstatement():
    if False:
        return 10

    def g():
        if False:
            while True:
                i = 10
        raise ValueError

    def f():
        if False:
            return 10
        g()
    try:
        f()
    except ValueError:
        excinfo = _pytest._code.ExceptionInfo.from_current()
    linenumbers = [f.__code__.co_firstlineno - 1 + 4, f.__code__.co_firstlineno - 1 + 1, g.__code__.co_firstlineno - 1 + 1]
    values = list(excinfo.traceback)
    foundlinenumbers = [x.lineno for x in values]
    assert foundlinenumbers == linenumbers

def f():
    if False:
        for i in range(10):
            print('nop')
    raise ValueError

def g():
    if False:
        i = 10
        return i + 15
    __tracebackhide__ = True
    f()

def h():
    if False:
        return 10
    g()

class TestTraceback_f_g_h:

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        try:
            h()
        except ValueError:
            self.excinfo = _pytest._code.ExceptionInfo.from_current()

    def test_traceback_entries(self):
        if False:
            print('Hello World!')
        tb = self.excinfo.traceback
        entries = list(tb)
        assert len(tb) == 4
        assert len(entries) == 4
        names = ['f', 'g', 'h']
        for entry in entries:
            try:
                names.remove(entry.frame.code.name)
            except ValueError:
                pass
        assert not names

    def test_traceback_entry_getsource(self):
        if False:
            return 10
        tb = self.excinfo.traceback
        s = str(tb[-1].getsource())
        assert s.startswith('def f():')
        assert s.endswith('raise ValueError')

    def test_traceback_entry_getsource_in_construct(self):
        if False:
            return 10

        def xyz():
            if False:
                while True:
                    i = 10
            try:
                raise ValueError
            except somenoname:
                pass
        try:
            xyz()
        except NameError:
            excinfo = _pytest._code.ExceptionInfo.from_current()
        else:
            assert False, 'did not raise NameError'
        tb = excinfo.traceback
        source = tb[-1].getsource()
        assert source is not None
        assert source.deindent().lines == ['def xyz():', '    try:', '        raise ValueError', '    except somenoname:  # type: ignore[name-defined] # noqa: F821']

    def test_traceback_cut(self) -> None:
        if False:
            while True:
                i = 10
        co = _pytest._code.Code.from_function(f)
        (path, firstlineno) = (co.path, co.firstlineno)
        assert isinstance(path, Path)
        traceback = self.excinfo.traceback
        newtraceback = traceback.cut(path=path, firstlineno=firstlineno)
        assert len(newtraceback) == 1
        newtraceback = traceback.cut(path=path, lineno=firstlineno + 2)
        assert len(newtraceback) == 1

    def test_traceback_cut_excludepath(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.makepyfile('def f(): raise ValueError')
        with pytest.raises(ValueError) as excinfo:
            import_path(p, root=pytester.path).f()
        basedir = Path(pytest.__file__).parent
        newtraceback = excinfo.traceback.cut(excludepath=basedir)
        for x in newtraceback:
            assert isinstance(x.path, Path)
            assert basedir not in x.path.parents
        assert newtraceback[-1].frame.code.path == p

    def test_traceback_filter(self):
        if False:
            i = 10
            return i + 15
        traceback = self.excinfo.traceback
        ntraceback = traceback.filter(self.excinfo)
        assert len(ntraceback) == len(traceback) - 1

    @pytest.mark.parametrize('tracebackhide, matching', [(lambda info: True, True), (lambda info: False, False), (operator.methodcaller('errisinstance', ValueError), True), (operator.methodcaller('errisinstance', IndexError), False)])
    def test_traceback_filter_selective(self, tracebackhide, matching):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                while True:
                    i = 10
            raise ValueError

        def g():
            if False:
                for i in range(10):
                    print('nop')
            __tracebackhide__ = tracebackhide
            f()

        def h():
            if False:
                print('Hello World!')
            g()
        excinfo = pytest.raises(ValueError, h)
        traceback = excinfo.traceback
        ntraceback = traceback.filter(excinfo)
        print(f'old: {traceback!r}')
        print(f'new: {ntraceback!r}')
        if matching:
            assert len(ntraceback) == len(traceback) - 2
        else:
            assert len(ntraceback) == len(traceback) - 1

    def test_traceback_recursion_index(self):
        if False:
            return 10

        def f(n):
            if False:
                while True:
                    i = 10
            if n < 10:
                n += 1
            f(n)
        excinfo = pytest.raises(RuntimeError, f, 8)
        traceback = excinfo.traceback
        recindex = traceback.recursionindex()
        assert recindex == 3

    def test_traceback_only_specific_recursion_errors(self, monkeypatch):
        if False:
            while True:
                i = 10

        def f(n):
            if False:
                for i in range(10):
                    print('nop')
            if n == 0:
                raise RuntimeError('hello')
            f(n - 1)
        excinfo = pytest.raises(RuntimeError, f, 25)
        monkeypatch.delattr(excinfo.traceback.__class__, 'recursionindex')
        repr = excinfo.getrepr()
        assert 'RuntimeError: hello' in str(repr.reprcrash)

    def test_traceback_no_recursion_index(self) -> None:
        if False:
            print('Hello World!')

        def do_stuff() -> None:
            if False:
                print('Hello World!')
            raise RuntimeError

        def reraise_me() -> None:
            if False:
                print('Hello World!')
            import sys
            (exc, val, tb) = sys.exc_info()
            assert val is not None
            raise val.with_traceback(tb)

        def f(n: int) -> None:
            if False:
                for i in range(10):
                    print('nop')
            try:
                do_stuff()
            except BaseException:
                reraise_me()
        excinfo = pytest.raises(RuntimeError, f, 8)
        assert excinfo is not None
        traceback = excinfo.traceback
        recindex = traceback.recursionindex()
        assert recindex is None

    def test_traceback_messy_recursion(self):
        if False:
            i = 10
            return i + 15
        decorator = pytest.importorskip('decorator').decorator

        def log(f, *k, **kw):
            if False:
                i = 10
                return i + 15
            print(f'{k} {kw}')
            f(*k, **kw)
        log = decorator(log)

        def fail():
            if False:
                while True:
                    i = 10
            raise ValueError('')
        fail = log(log(fail))
        excinfo = pytest.raises(ValueError, fail)
        assert excinfo.traceback.recursionindex() is None

    def test_getreprcrash(self):
        if False:
            return 10

        def i():
            if False:
                i = 10
                return i + 15
            __tracebackhide__ = True
            raise ValueError

        def h():
            if False:
                while True:
                    i = 10
            i()

        def g():
            if False:
                i = 10
                return i + 15
            __tracebackhide__ = True
            h()

        def f():
            if False:
                while True:
                    i = 10
            g()
        excinfo = pytest.raises(ValueError, f)
        reprcrash = excinfo._getreprcrash()
        assert reprcrash is not None
        co = _pytest._code.Code.from_function(h)
        assert reprcrash.path == str(co.path)
        assert reprcrash.lineno == co.firstlineno + 1 + 1

    def test_getreprcrash_empty(self):
        if False:
            for i in range(10):
                print('nop')

        def g():
            if False:
                i = 10
                return i + 15
            __tracebackhide__ = True
            raise ValueError

        def f():
            if False:
                return 10
            __tracebackhide__ = True
            g()
        excinfo = pytest.raises(ValueError, f)
        assert excinfo._getreprcrash() is None

def test_excinfo_exconly():
    if False:
        for i in range(10):
            print('nop')
    excinfo = pytest.raises(ValueError, h)
    assert excinfo.exconly().startswith('ValueError')
    with pytest.raises(ValueError) as excinfo:
        raise ValueError('hello\nworld')
    msg = excinfo.exconly(tryshort=True)
    assert msg.startswith('ValueError')
    assert msg.endswith('world')

def test_excinfo_repr_str() -> None:
    if False:
        print('Hello World!')
    excinfo1 = pytest.raises(ValueError, h)
    assert repr(excinfo1) == '<ExceptionInfo ValueError() tblen=4>'
    assert str(excinfo1) == '<ExceptionInfo ValueError() tblen=4>'

    class CustomException(Exception):

        def __repr__(self):
            if False:
                return 10
            return 'custom_repr'

    def raises() -> None:
        if False:
            print('Hello World!')
        raise CustomException()
    excinfo2 = pytest.raises(CustomException, raises)
    assert repr(excinfo2) == '<ExceptionInfo custom_repr tblen=2>'
    assert str(excinfo2) == '<ExceptionInfo custom_repr tblen=2>'

def test_excinfo_for_later() -> None:
    if False:
        while True:
            i = 10
    e = ExceptionInfo[BaseException].for_later()
    assert 'for raises' in repr(e)
    assert 'for raises' in str(e)

def test_excinfo_errisinstance():
    if False:
        for i in range(10):
            print('nop')
    excinfo = pytest.raises(ValueError, h)
    assert excinfo.errisinstance(ValueError)

def test_excinfo_no_sourcecode():
    if False:
        print('Hello World!')
    try:
        exec('raise ValueError()')
    except ValueError:
        excinfo = _pytest._code.ExceptionInfo.from_current()
    s = str(excinfo.traceback[-1])
    assert s == "  File '<string>':1 in <module>\n  ???\n"

def test_excinfo_no_python_sourcecode(tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    tmp_path.joinpath('test.txt').write_text('{{ h()}}:', encoding='utf-8')
    jinja2 = pytest.importorskip('jinja2')
    loader = jinja2.FileSystemLoader(str(tmp_path))
    env = jinja2.Environment(loader=loader)
    template = env.get_template('test.txt')
    excinfo = pytest.raises(ValueError, template.render, h=h)
    for item in excinfo.traceback:
        print(item)
        item.source
        if isinstance(item.path, Path) and item.path.name == 'test.txt':
            assert str(item.source) == '{{ h()}}:'

def test_entrysource_Queue_example():
    if False:
        print('Hello World!')
    try:
        queue.Queue().get(timeout=0.001)
    except queue.Empty:
        excinfo = _pytest._code.ExceptionInfo.from_current()
    entry = excinfo.traceback[-1]
    source = entry.getsource()
    assert source is not None
    s = str(source).strip()
    assert s.startswith('def get')

def test_codepath_Queue_example() -> None:
    if False:
        return 10
    try:
        queue.Queue().get(timeout=0.001)
    except queue.Empty:
        excinfo = _pytest._code.ExceptionInfo.from_current()
    entry = excinfo.traceback[-1]
    path = entry.path
    assert isinstance(path, Path)
    assert path.name.lower() == 'queue.py'
    assert path.exists()

def test_match_succeeds():
    if False:
        return 10
    with pytest.raises(ZeroDivisionError) as excinfo:
        0 // 0
    excinfo.match('.*zero.*')

def test_match_raises_error(pytester: Pytester) -> None:
    if False:
        return 10
    pytester.makepyfile("\n        import pytest\n        def test_division_zero():\n            with pytest.raises(ZeroDivisionError) as excinfo:\n                0 / 0\n            excinfo.match(r'[123]+')\n    ")
    result = pytester.runpytest('--tb=short')
    assert result.ret != 0
    match = ['E .* AssertionError: Regex pattern did not match.', "E .* Regex: '\\[123\\]\\+'", "E .* Input: 'division by zero'"]
    result.stdout.re_match_lines(match)
    result.stdout.no_fnmatch_line('*__tracebackhide__ = True*')
    result = pytester.runpytest('--fulltrace')
    assert result.ret != 0
    result.stdout.re_match_lines(['.*__tracebackhide__ = True.*', *match])

class TestGroupContains:

    def test_contains_exception_type(self) -> None:
        if False:
            while True:
                i = 10
        exc_group = ExceptionGroup('', [RuntimeError()])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError)

    def test_doesnt_contain_exception_type(self) -> None:
        if False:
            while True:
                i = 10
        exc_group = ExceptionGroup('', [ValueError()])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert not exc_info.group_contains(RuntimeError)

    def test_contains_exception_match(self) -> None:
        if False:
            print('Hello World!')
        exc_group = ExceptionGroup('', [RuntimeError('exception message')])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, match='^exception message$')

    def test_doesnt_contain_exception_match(self) -> None:
        if False:
            while True:
                i = 10
        exc_group = ExceptionGroup('', [RuntimeError('message that will not match')])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert not exc_info.group_contains(RuntimeError, match='^exception message$')

    def test_contains_exception_type_unlimited_depth(self) -> None:
        if False:
            print('Hello World!')
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError()])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError)

    def test_contains_exception_type_at_depth_1(self) -> None:
        if False:
            while True:
                i = 10
        exc_group = ExceptionGroup('', [RuntimeError()])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, depth=1)

    def test_doesnt_contain_exception_type_past_depth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError()])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert not exc_info.group_contains(RuntimeError, depth=1)

    def test_contains_exception_type_specific_depth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError()])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, depth=2)

    def test_contains_exception_match_unlimited_depth(self) -> None:
        if False:
            return 10
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError('exception message')])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, match='^exception message$')

    def test_contains_exception_match_at_depth_1(self) -> None:
        if False:
            i = 10
            return i + 15
        exc_group = ExceptionGroup('', [RuntimeError('exception message')])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, match='^exception message$', depth=1)

    def test_doesnt_contain_exception_match_past_depth(self) -> None:
        if False:
            print('Hello World!')
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError('exception message')])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert not exc_info.group_contains(RuntimeError, match='^exception message$', depth=1)

    def test_contains_exception_match_specific_depth(self) -> None:
        if False:
            return 10
        exc_group = ExceptionGroup('', [ExceptionGroup('', [RuntimeError('exception message')])])
        with pytest.raises(ExceptionGroup) as exc_info:
            raise exc_group
        assert exc_info.group_contains(RuntimeError, match='^exception message$', depth=2)

class TestFormattedExcinfo:

    @pytest.fixture
    def importasmod(self, tmp_path: Path, _sys_snapshot):
        if False:
            i = 10
            return i + 15

        def importasmod(source):
            if False:
                while True:
                    i = 10
            source = textwrap.dedent(source)
            modpath = tmp_path.joinpath('mod.py')
            tmp_path.joinpath('__init__.py').touch()
            modpath.write_text(source, encoding='utf-8')
            importlib.invalidate_caches()
            return import_path(modpath, root=tmp_path)
        return importasmod

    def test_repr_source(self):
        if False:
            return 10
        pr = FormattedExcinfo()
        source = _pytest._code.Source('            def f(x):\n                pass\n            ').strip()
        pr.flow_marker = '|'
        lines = pr.get_source(source, 0)
        assert len(lines) == 2
        assert lines[0] == '|   def f(x):'
        assert lines[1] == '        pass'

    def test_repr_source_out_of_bounds(self):
        if False:
            while True:
                i = 10
        pr = FormattedExcinfo()
        source = _pytest._code.Source('            def f(x):\n                pass\n            ').strip()
        pr.flow_marker = '|'
        lines = pr.get_source(source, 100)
        assert len(lines) == 1
        assert lines[0] == '|   ???'
        lines = pr.get_source(source, -100)
        assert len(lines) == 1
        assert lines[0] == '|   ???'

    def test_repr_source_excinfo(self) -> None:
        if False:
            return 10
        'Check if indentation is right.'
        try:

            def f():
                if False:
                    i = 10
                    return i + 15
                1 / 0
            f()
        except BaseException:
            excinfo = _pytest._code.ExceptionInfo.from_current()
        else:
            assert False, 'did not raise'
        pr = FormattedExcinfo()
        source = pr._getentrysource(excinfo.traceback[-1])
        assert source is not None
        lines = pr.get_source(source, 1, excinfo)
        for line in lines:
            print(line)
        assert lines == ['    def f():', '>       1 / 0', 'E       ZeroDivisionError: division by zero']

    def test_repr_source_not_existing(self):
        if False:
            return 10
        pr = FormattedExcinfo()
        co = compile('raise ValueError()', '', 'exec')
        try:
            exec(co)
        except ValueError:
            excinfo = _pytest._code.ExceptionInfo.from_current()
        repr = pr.repr_excinfo(excinfo)
        assert repr.reprtraceback.reprentries[1].lines[0] == '>   ???'
        assert repr.chain[0][0].reprentries[1].lines[0] == '>   ???'

    def test_repr_many_line_source_not_existing(self):
        if False:
            i = 10
            return i + 15
        pr = FormattedExcinfo()
        co = compile('\na = 1\nraise ValueError()\n', '', 'exec')
        try:
            exec(co)
        except ValueError:
            excinfo = _pytest._code.ExceptionInfo.from_current()
        repr = pr.repr_excinfo(excinfo)
        assert repr.reprtraceback.reprentries[1].lines[0] == '>   ???'
        assert repr.chain[0][0].reprentries[1].lines[0] == '>   ???'

    def test_repr_source_failing_fullsource(self, monkeypatch) -> None:
        if False:
            while True:
                i = 10
        pr = FormattedExcinfo()
        try:
            1 / 0
        except ZeroDivisionError:
            excinfo = ExceptionInfo.from_current()
        with monkeypatch.context() as m:
            m.setattr(_pytest._code.Code, 'fullsource', property(lambda self: None))
            repr = pr.repr_excinfo(excinfo)
        assert repr.reprtraceback.reprentries[0].lines[0] == '>   ???'
        assert repr.chain[0][0].reprentries[0].lines[0] == '>   ???'

    def test_repr_local(self) -> None:
        if False:
            return 10
        p = FormattedExcinfo(showlocals=True)
        loc = {'y': 5, 'z': 7, 'x': 3, '@x': 2, '__builtins__': {}}
        reprlocals = p.repr_locals(loc)
        assert reprlocals is not None
        assert reprlocals.lines
        assert reprlocals.lines[0] == '__builtins__ = <builtins>'
        assert reprlocals.lines[1] == 'x          = 3'
        assert reprlocals.lines[2] == 'y          = 5'
        assert reprlocals.lines[3] == 'z          = 7'

    def test_repr_local_with_error(self) -> None:
        if False:
            print('Hello World!')

        class ObjWithErrorInRepr:

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                raise NotImplementedError
        p = FormattedExcinfo(showlocals=True, truncate_locals=False)
        loc = {'x': ObjWithErrorInRepr(), '__builtins__': {}}
        reprlocals = p.repr_locals(loc)
        assert reprlocals is not None
        assert reprlocals.lines
        assert reprlocals.lines[0] == '__builtins__ = <builtins>'
        assert '[NotImplementedError() raised in repr()]' in reprlocals.lines[1]

    def test_repr_local_with_exception_in_class_property(self) -> None:
        if False:
            print('Hello World!')

        class ExceptionWithBrokenClass(Exception):

            @property
            def __class__(self):
                if False:
                    return 10
                raise TypeError('boom!')

        class ObjWithErrorInRepr:

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                raise ExceptionWithBrokenClass()
        p = FormattedExcinfo(showlocals=True, truncate_locals=False)
        loc = {'x': ObjWithErrorInRepr(), '__builtins__': {}}
        reprlocals = p.repr_locals(loc)
        assert reprlocals is not None
        assert reprlocals.lines
        assert reprlocals.lines[0] == '__builtins__ = <builtins>'
        assert '[ExceptionWithBrokenClass() raised in repr()]' in reprlocals.lines[1]

    def test_repr_local_truncated(self) -> None:
        if False:
            i = 10
            return i + 15
        loc = {'l': [i for i in range(10)]}
        p = FormattedExcinfo(showlocals=True)
        truncated_reprlocals = p.repr_locals(loc)
        assert truncated_reprlocals is not None
        assert truncated_reprlocals.lines
        assert truncated_reprlocals.lines[0] == 'l          = [0, 1, 2, 3, 4, 5, ...]'
        q = FormattedExcinfo(showlocals=True, truncate_locals=False)
        full_reprlocals = q.repr_locals(loc)
        assert full_reprlocals is not None
        assert full_reprlocals.lines
        assert full_reprlocals.lines[0] == 'l          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'

    def test_repr_tracebackentry_lines(self, importasmod) -> None:
        if False:
            return 10
        mod = importasmod('\n            def func1():\n                raise ValueError("hello\\nworld")\n        ')
        excinfo = pytest.raises(ValueError, mod.func1)
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        p = FormattedExcinfo()
        reprtb = p.repr_traceback_entry(excinfo.traceback[-1])
        lines = reprtb.lines
        assert lines[0] == '    def func1():'
        assert lines[1] == '>       raise ValueError("hello\\nworld")'
        p = FormattedExcinfo(showlocals=True)
        repr_entry = p.repr_traceback_entry(excinfo.traceback[-1], excinfo)
        lines = repr_entry.lines
        assert lines[0] == '    def func1():'
        assert lines[1] == '>       raise ValueError("hello\\nworld")'
        assert lines[2] == 'E       ValueError: hello'
        assert lines[3] == 'E       world'
        assert not lines[4:]
        loc = repr_entry.reprfileloc
        assert loc is not None
        assert loc.path == mod.__file__
        assert loc.lineno == 3

    def test_repr_tracebackentry_lines2(self, importasmod, tw_mock) -> None:
        if False:
            i = 10
            return i + 15
        mod = importasmod('\n            def func1(m, x, y, z):\n                raise ValueError("hello\\nworld")\n        ')
        excinfo = pytest.raises(ValueError, mod.func1, 'm' * 90, 5, 13, 'z' * 120)
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        entry = excinfo.traceback[-1]
        p = FormattedExcinfo(funcargs=True)
        reprfuncargs = p.repr_args(entry)
        assert reprfuncargs is not None
        assert reprfuncargs.args[0] == ('m', repr('m' * 90))
        assert reprfuncargs.args[1] == ('x', '5')
        assert reprfuncargs.args[2] == ('y', '13')
        assert reprfuncargs.args[3] == ('z', repr('z' * 120))
        p = FormattedExcinfo(funcargs=True)
        repr_entry = p.repr_traceback_entry(entry)
        assert repr_entry.reprfuncargs is not None
        assert repr_entry.reprfuncargs.args == reprfuncargs.args
        repr_entry.toterminal(tw_mock)
        assert tw_mock.lines[0] == 'm = ' + repr('m' * 90)
        assert tw_mock.lines[1] == 'x = 5, y = 13'
        assert tw_mock.lines[2] == 'z = ' + repr('z' * 120)

    def test_repr_tracebackentry_lines_var_kw_args(self, importasmod, tw_mock) -> None:
        if False:
            i = 10
            return i + 15
        mod = importasmod('\n            def func1(x, *y, **z):\n                raise ValueError("hello\\nworld")\n        ')
        excinfo = pytest.raises(ValueError, mod.func1, 'a', 'b', c='d')
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        entry = excinfo.traceback[-1]
        p = FormattedExcinfo(funcargs=True)
        reprfuncargs = p.repr_args(entry)
        assert reprfuncargs is not None
        assert reprfuncargs.args[0] == ('x', repr('a'))
        assert reprfuncargs.args[1] == ('y', repr(('b',)))
        assert reprfuncargs.args[2] == ('z', repr({'c': 'd'}))
        p = FormattedExcinfo(funcargs=True)
        repr_entry = p.repr_traceback_entry(entry)
        assert repr_entry.reprfuncargs
        assert repr_entry.reprfuncargs.args == reprfuncargs.args
        repr_entry.toterminal(tw_mock)
        assert tw_mock.lines[0] == "x = 'a', y = ('b',), z = {'c': 'd'}"

    def test_repr_tracebackentry_short(self, importasmod) -> None:
        if False:
            print('Hello World!')
        mod = importasmod('\n            def func1():\n                raise ValueError("hello")\n            def entry():\n                func1()\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        p = FormattedExcinfo(style='short')
        reprtb = p.repr_traceback_entry(excinfo.traceback[-2])
        lines = reprtb.lines
        basename = Path(mod.__file__).name
        assert lines[0] == '    func1()'
        assert reprtb.reprfileloc is not None
        assert basename in str(reprtb.reprfileloc.path)
        assert reprtb.reprfileloc.lineno == 5
        p = FormattedExcinfo(style='short')
        reprtb = p.repr_traceback_entry(excinfo.traceback[-1], excinfo)
        lines = reprtb.lines
        assert lines[0] == '    raise ValueError("hello")'
        assert lines[1] == 'E   ValueError: hello'
        assert reprtb.reprfileloc is not None
        assert basename in str(reprtb.reprfileloc.path)
        assert reprtb.reprfileloc.lineno == 3

    def test_repr_tracebackentry_no(self, importasmod):
        if False:
            i = 10
            return i + 15
        mod = importasmod('\n            def func1():\n                raise ValueError("hello")\n            def entry():\n                func1()\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        p = FormattedExcinfo(style='no')
        p.repr_traceback_entry(excinfo.traceback[-2])
        p = FormattedExcinfo(style='no')
        reprentry = p.repr_traceback_entry(excinfo.traceback[-1], excinfo)
        lines = reprentry.lines
        assert lines[0] == 'E   ValueError: hello'
        assert not lines[1:]

    def test_repr_traceback_tbfilter(self, importasmod):
        if False:
            for i in range(10):
                print('nop')
        mod = importasmod('\n            def f(x):\n                raise ValueError(x)\n            def entry():\n                f(0)\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        p = FormattedExcinfo(tbfilter=True)
        reprtb = p.repr_traceback(excinfo)
        assert len(reprtb.reprentries) == 2
        p = FormattedExcinfo(tbfilter=False)
        reprtb = p.repr_traceback(excinfo)
        assert len(reprtb.reprentries) == 3

    def test_traceback_short_no_source(self, importasmod, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        mod = importasmod('\n            def func1():\n                raise ValueError("hello")\n            def entry():\n                func1()\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        from _pytest._code.code import Code
        with monkeypatch.context() as mp:
            mp.setattr(Code, 'path', 'bogus')
            p = FormattedExcinfo(style='short')
            reprtb = p.repr_traceback_entry(excinfo.traceback[-2])
            lines = reprtb.lines
            last_p = FormattedExcinfo(style='short')
            last_reprtb = last_p.repr_traceback_entry(excinfo.traceback[-1], excinfo)
            last_lines = last_reprtb.lines
        assert lines[0] == '    func1()'
        assert last_lines[0] == '    raise ValueError("hello")'
        assert last_lines[1] == 'E   ValueError: hello'

    def test_repr_traceback_and_excinfo(self, importasmod) -> None:
        if False:
            return 10
        mod = importasmod('\n            def f(x):\n                raise ValueError(x)\n            def entry():\n                f(0)\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        styles: tuple[_TracebackStyle, ...] = ('long', 'short')
        for style in styles:
            p = FormattedExcinfo(style=style)
            reprtb = p.repr_traceback(excinfo)
            assert len(reprtb.reprentries) == 2
            assert reprtb.style == style
            assert not reprtb.extraline
            repr = p.repr_excinfo(excinfo)
            assert repr.reprtraceback
            assert len(repr.reprtraceback.reprentries) == len(reprtb.reprentries)
            assert repr.chain[0][0]
            assert len(repr.chain[0][0].reprentries) == len(reprtb.reprentries)
            assert repr.reprcrash is not None
            assert repr.reprcrash.path.endswith('mod.py')
            assert repr.reprcrash.message == 'ValueError: 0'

    def test_repr_traceback_with_invalid_cwd(self, importasmod, monkeypatch) -> None:
        if False:
            print('Hello World!')
        mod = importasmod('\n            def f(x):\n                raise ValueError(x)\n            def entry():\n                f(0)\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        p = FormattedExcinfo(abspath=False)
        raised = 0
        orig_path_cwd = Path.cwd

        def raiseos():
            if False:
                i = 10
                return i + 15
            nonlocal raised
            upframe = sys._getframe().f_back
            assert upframe is not None
            if upframe.f_code.co_name == '_makepath':
                raised += 1
                raise OSError(2, 'custom_oserror')
            return orig_path_cwd()
        monkeypatch.setattr(Path, 'cwd', raiseos)
        assert p._makepath(Path(__file__)) == __file__
        assert raised == 1
        repr_tb = p.repr_traceback(excinfo)
        matcher = LineMatcher(str(repr_tb).splitlines())
        matcher.fnmatch_lines(['def entry():', '>       f(0)', '', f'{mod.__file__}:5: ', '_ _ *', '', '    def f(x):', '>       raise ValueError(x)', 'E       ValueError: 0', '', f'{mod.__file__}:3: ValueError'])
        assert raised == 3

    def test_repr_excinfo_addouterr(self, importasmod, tw_mock):
        if False:
            return 10
        mod = importasmod('\n            def entry():\n                raise ValueError()\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        repr = excinfo.getrepr()
        repr.addsection('title', 'content')
        repr.toterminal(tw_mock)
        assert tw_mock.lines[-1] == 'content'
        assert tw_mock.lines[-2] == ('-', 'title')

    def test_repr_excinfo_reprcrash(self, importasmod) -> None:
        if False:
            print('Hello World!')
        mod = importasmod('\n            def entry():\n                raise ValueError()\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        repr = excinfo.getrepr()
        assert repr.reprcrash is not None
        assert repr.reprcrash.path.endswith('mod.py')
        assert repr.reprcrash.lineno == 3
        assert repr.reprcrash.message == 'ValueError'
        assert str(repr.reprcrash).endswith('mod.py:3: ValueError')

    def test_repr_traceback_recursion(self, importasmod):
        if False:
            return 10
        mod = importasmod('\n            def rec2(x):\n                return rec1(x+1)\n            def rec1(x):\n                return rec2(x-1)\n            def entry():\n                rec1(42)\n        ')
        excinfo = pytest.raises(RuntimeError, mod.entry)
        for style in ('short', 'long', 'no'):
            p = FormattedExcinfo(style='short')
            reprtb = p.repr_traceback(excinfo)
            assert reprtb.extraline == '!!! Recursion detected (same locals & position)'
            assert str(reprtb)

    def test_reprexcinfo_getrepr(self, importasmod) -> None:
        if False:
            i = 10
            return i + 15
        mod = importasmod('\n            def f(x):\n                raise ValueError(x)\n            def entry():\n                f(0)\n        ')
        excinfo = pytest.raises(ValueError, mod.entry)
        styles: tuple[_TracebackStyle, ...] = ('short', 'long', 'no')
        for style in styles:
            for showlocals in (True, False):
                repr = excinfo.getrepr(style=style, showlocals=showlocals)
                assert repr.reprtraceback.style == style
                assert isinstance(repr, ExceptionChainRepr)
                for r in repr.chain:
                    assert r[0].style == style

    def test_reprexcinfo_unicode(self):
        if False:
            print('Hello World!')
        from _pytest._code.code import TerminalRepr

        class MyRepr(TerminalRepr):

            def toterminal(self, tw: TerminalWriter) -> None:
                if False:
                    i = 10
                    return i + 15
                tw.line('я')
        x = str(MyRepr())
        assert x == 'я'

    def test_toterminal_long(self, importasmod, tw_mock):
        if False:
            for i in range(10):
                print('nop')
        mod = importasmod('\n            def g(x):\n                raise ValueError(x)\n            def f():\n                g(3)\n        ')
        excinfo = pytest.raises(ValueError, mod.f)
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        repr = excinfo.getrepr()
        repr.toterminal(tw_mock)
        assert tw_mock.lines[0] == ''
        tw_mock.lines.pop(0)
        assert tw_mock.lines[0] == '    def f():'
        assert tw_mock.lines[1] == '>       g(3)'
        assert tw_mock.lines[2] == ''
        line = tw_mock.get_write_msg(3)
        assert line.endswith('mod.py')
        assert tw_mock.lines[4] == ':5: '
        assert tw_mock.lines[5] == ('_ ', None)
        assert tw_mock.lines[6] == ''
        assert tw_mock.lines[7] == '    def g(x):'
        assert tw_mock.lines[8] == '>       raise ValueError(x)'
        assert tw_mock.lines[9] == 'E       ValueError: 3'
        assert tw_mock.lines[10] == ''
        line = tw_mock.get_write_msg(11)
        assert line.endswith('mod.py')
        assert tw_mock.lines[12] == ':3: ValueError'

    def test_toterminal_long_missing_source(self, importasmod, tmp_path: Path, tw_mock) -> None:
        if False:
            print('Hello World!')
        mod = importasmod('\n            def g(x):\n                raise ValueError(x)\n            def f():\n                g(3)\n        ')
        excinfo = pytest.raises(ValueError, mod.f)
        tmp_path.joinpath('mod.py').unlink()
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        repr = excinfo.getrepr()
        repr.toterminal(tw_mock)
        assert tw_mock.lines[0] == ''
        tw_mock.lines.pop(0)
        assert tw_mock.lines[0] == '>   ???'
        assert tw_mock.lines[1] == ''
        line = tw_mock.get_write_msg(2)
        assert line.endswith('mod.py')
        assert tw_mock.lines[3] == ':5: '
        assert tw_mock.lines[4] == ('_ ', None)
        assert tw_mock.lines[5] == ''
        assert tw_mock.lines[6] == '>   ???'
        assert tw_mock.lines[7] == 'E   ValueError: 3'
        assert tw_mock.lines[8] == ''
        line = tw_mock.get_write_msg(9)
        assert line.endswith('mod.py')
        assert tw_mock.lines[10] == ':3: ValueError'

    def test_toterminal_long_incomplete_source(self, importasmod, tmp_path: Path, tw_mock) -> None:
        if False:
            for i in range(10):
                print('nop')
        mod = importasmod('\n            def g(x):\n                raise ValueError(x)\n            def f():\n                g(3)\n        ')
        excinfo = pytest.raises(ValueError, mod.f)
        tmp_path.joinpath('mod.py').write_text('asdf', encoding='utf-8')
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        repr = excinfo.getrepr()
        repr.toterminal(tw_mock)
        assert tw_mock.lines[0] == ''
        tw_mock.lines.pop(0)
        assert tw_mock.lines[0] == '>   ???'
        assert tw_mock.lines[1] == ''
        line = tw_mock.get_write_msg(2)
        assert line.endswith('mod.py')
        assert tw_mock.lines[3] == ':5: '
        assert tw_mock.lines[4] == ('_ ', None)
        assert tw_mock.lines[5] == ''
        assert tw_mock.lines[6] == '>   ???'
        assert tw_mock.lines[7] == 'E   ValueError: 3'
        assert tw_mock.lines[8] == ''
        line = tw_mock.get_write_msg(9)
        assert line.endswith('mod.py')
        assert tw_mock.lines[10] == ':3: ValueError'

    def test_toterminal_long_filenames(self, importasmod, tw_mock, monkeypatch: MonkeyPatch) -> None:
        if False:
            i = 10
            return i + 15
        mod = importasmod('\n            def f():\n                raise ValueError()\n        ')
        excinfo = pytest.raises(ValueError, mod.f)
        path = Path(mod.__file__)
        monkeypatch.chdir(path.parent)
        repr = excinfo.getrepr(abspath=False)
        repr.toterminal(tw_mock)
        x = bestrelpath(Path.cwd(), path)
        if len(x) < len(str(path)):
            msg = tw_mock.get_write_msg(-2)
            assert msg == 'mod.py'
            assert tw_mock.lines[-1] == ':3: ValueError'
        repr = excinfo.getrepr(abspath=True)
        repr.toterminal(tw_mock)
        msg = tw_mock.get_write_msg(-2)
        assert msg == str(path)
        line = tw_mock.lines[-1]
        assert line == ':3: ValueError'

    @pytest.mark.parametrize('reproptions', [pytest.param({'style': style, 'showlocals': showlocals, 'funcargs': funcargs, 'tbfilter': tbfilter}, id='style={},showlocals={},funcargs={},tbfilter={}'.format(style, showlocals, funcargs, tbfilter)) for style in ['long', 'short', 'line', 'no', 'native', 'value', 'auto'] for showlocals in (True, False) for tbfilter in (True, False) for funcargs in (True, False)])
    def test_format_excinfo(self, reproptions: dict[str, Any]) -> None:
        if False:
            return 10

        def bar():
            if False:
                return 10
            assert False, 'some error'

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            bar()
        with pytest.raises(AssertionError) as excinfo:
            foo()
        file = io.StringIO()
        tw = TerminalWriter(file=file)
        repr = excinfo.getrepr(**reproptions)
        repr.toterminal(tw)
        assert file.getvalue()

    def test_traceback_repr_style(self, importasmod, tw_mock):
        if False:
            return 10
        mod = importasmod('\n            def f():\n                g()\n            def g():\n                h()\n            def h():\n                i()\n            def i():\n                raise ValueError()\n        ')
        excinfo = pytest.raises(ValueError, mod.f)
        excinfo.traceback = excinfo.traceback.filter(excinfo)
        excinfo.traceback = _pytest._code.Traceback((entry if i not in (1, 2) else entry.with_repr_style('short') for (i, entry) in enumerate(excinfo.traceback)))
        r = excinfo.getrepr(style='long')
        r.toterminal(tw_mock)
        for line in tw_mock.lines:
            print(line)
        assert tw_mock.lines[0] == ''
        assert tw_mock.lines[1] == '    def f():'
        assert tw_mock.lines[2] == '>       g()'
        assert tw_mock.lines[3] == ''
        msg = tw_mock.get_write_msg(4)
        assert msg.endswith('mod.py')
        assert tw_mock.lines[5] == ':3: '
        assert tw_mock.lines[6] == ('_ ', None)
        tw_mock.get_write_msg(7)
        assert tw_mock.lines[8].endswith('in g')
        assert tw_mock.lines[9] == '    h()'
        tw_mock.get_write_msg(10)
        assert tw_mock.lines[11].endswith('in h')
        assert tw_mock.lines[12] == '    i()'
        assert tw_mock.lines[13] == ('_ ', None)
        assert tw_mock.lines[14] == ''
        assert tw_mock.lines[15] == '    def i():'
        assert tw_mock.lines[16] == '>       raise ValueError()'
        assert tw_mock.lines[17] == 'E       ValueError'
        assert tw_mock.lines[18] == ''
        msg = tw_mock.get_write_msg(19)
        msg.endswith('mod.py')
        assert tw_mock.lines[20] == ':9: ValueError'

    def test_exc_chain_repr(self, importasmod, tw_mock):
        if False:
            print('Hello World!')
        mod = importasmod('\n            class Err(Exception):\n                pass\n            def f():\n                try:\n                    g()\n                except Exception as e:\n                    raise Err() from e\n                finally:\n                    h()\n            def g():\n                raise ValueError()\n\n            def h():\n                raise AttributeError()\n        ')
        excinfo = pytest.raises(AttributeError, mod.f)
        r = excinfo.getrepr(style='long')
        r.toterminal(tw_mock)
        for line in tw_mock.lines:
            print(line)
        assert tw_mock.lines[0] == ''
        assert tw_mock.lines[1] == '    def f():'
        assert tw_mock.lines[2] == '        try:'
        assert tw_mock.lines[3] == '>           g()'
        assert tw_mock.lines[4] == ''
        line = tw_mock.get_write_msg(5)
        assert line.endswith('mod.py')
        assert tw_mock.lines[6] == ':6: '
        assert tw_mock.lines[7] == ('_ ', None)
        assert tw_mock.lines[8] == ''
        assert tw_mock.lines[9] == '    def g():'
        assert tw_mock.lines[10] == '>       raise ValueError()'
        assert tw_mock.lines[11] == 'E       ValueError'
        assert tw_mock.lines[12] == ''
        line = tw_mock.get_write_msg(13)
        assert line.endswith('mod.py')
        assert tw_mock.lines[14] == ':12: ValueError'
        assert tw_mock.lines[15] == ''
        assert tw_mock.lines[16] == 'The above exception was the direct cause of the following exception:'
        assert tw_mock.lines[17] == ''
        assert tw_mock.lines[18] == '    def f():'
        assert tw_mock.lines[19] == '        try:'
        assert tw_mock.lines[20] == '            g()'
        assert tw_mock.lines[21] == '        except Exception as e:'
        assert tw_mock.lines[22] == '>           raise Err() from e'
        assert tw_mock.lines[23] == 'E           test_exc_chain_repr0.mod.Err'
        assert tw_mock.lines[24] == ''
        line = tw_mock.get_write_msg(25)
        assert line.endswith('mod.py')
        assert tw_mock.lines[26] == ':8: Err'
        assert tw_mock.lines[27] == ''
        assert tw_mock.lines[28] == 'During handling of the above exception, another exception occurred:'
        assert tw_mock.lines[29] == ''
        assert tw_mock.lines[30] == '    def f():'
        assert tw_mock.lines[31] == '        try:'
        assert tw_mock.lines[32] == '            g()'
        assert tw_mock.lines[33] == '        except Exception as e:'
        assert tw_mock.lines[34] == '            raise Err() from e'
        assert tw_mock.lines[35] == '        finally:'
        assert tw_mock.lines[36] == '>           h()'
        assert tw_mock.lines[37] == ''
        line = tw_mock.get_write_msg(38)
        assert line.endswith('mod.py')
        assert tw_mock.lines[39] == ':10: '
        assert tw_mock.lines[40] == ('_ ', None)
        assert tw_mock.lines[41] == ''
        assert tw_mock.lines[42] == '    def h():'
        assert tw_mock.lines[43] == '>       raise AttributeError()'
        assert tw_mock.lines[44] == 'E       AttributeError'
        assert tw_mock.lines[45] == ''
        line = tw_mock.get_write_msg(46)
        assert line.endswith('mod.py')
        assert tw_mock.lines[47] == ':15: AttributeError'

    @pytest.mark.parametrize('mode', ['from_none', 'explicit_suppress'])
    def test_exc_repr_chain_suppression(self, importasmod, mode, tw_mock):
        if False:
            i = 10
            return i + 15
        'Check that exc repr does not show chained exceptions in Python 3.\n        - When the exception is raised with "from None"\n        - Explicitly suppressed with "chain=False" to ExceptionInfo.getrepr().\n        '
        raise_suffix = ' from None' if mode == 'from_none' else ''
        mod = importasmod('\n            def f():\n                try:\n                    g()\n                except Exception:\n                    raise AttributeError(){raise_suffix}\n            def g():\n                raise ValueError()\n        '.format(raise_suffix=raise_suffix))
        excinfo = pytest.raises(AttributeError, mod.f)
        r = excinfo.getrepr(style='long', chain=mode != 'explicit_suppress')
        r.toterminal(tw_mock)
        for line in tw_mock.lines:
            print(line)
        assert tw_mock.lines[0] == ''
        assert tw_mock.lines[1] == '    def f():'
        assert tw_mock.lines[2] == '        try:'
        assert tw_mock.lines[3] == '            g()'
        assert tw_mock.lines[4] == '        except Exception:'
        assert tw_mock.lines[5] == '>           raise AttributeError(){}'.format(raise_suffix)
        assert tw_mock.lines[6] == 'E           AttributeError'
        assert tw_mock.lines[7] == ''
        line = tw_mock.get_write_msg(8)
        assert line.endswith('mod.py')
        assert tw_mock.lines[9] == ':6: AttributeError'
        assert len(tw_mock.lines) == 10

    @pytest.mark.parametrize('reason, description', [pytest.param('cause', 'The above exception was the direct cause of the following exception:', id='cause'), pytest.param('context', 'During handling of the above exception, another exception occurred:', id='context')])
    def test_exc_chain_repr_without_traceback(self, importasmod, reason, description):
        if False:
            return 10
        "\n        Handle representation of exception chains where one of the exceptions doesn't have a\n        real traceback, such as those raised in a subprocess submitted by the multiprocessing\n        module (#1984).\n        "
        exc_handling_code = ' from e' if reason == 'cause' else ''
        mod = importasmod("\n            def f():\n                try:\n                    g()\n                except Exception as e:\n                    raise RuntimeError('runtime problem'){exc_handling_code}\n            def g():\n                raise ValueError('invalid value')\n        ".format(exc_handling_code=exc_handling_code))
        with pytest.raises(RuntimeError) as excinfo:
            mod.f()
        attr = '__%s__' % reason
        getattr(excinfo.value, attr).__traceback__ = None
        r = excinfo.getrepr()
        file = io.StringIO()
        tw = TerminalWriter(file=file)
        tw.hasmarkup = False
        r.toterminal(tw)
        matcher = LineMatcher(file.getvalue().splitlines())
        matcher.fnmatch_lines(['ValueError: invalid value', description, '* except Exception as e:', "> * raise RuntimeError('runtime problem')" + exc_handling_code, 'E *RuntimeError: runtime problem'])

    def test_exc_chain_repr_cycle(self, importasmod, tw_mock):
        if False:
            return 10
        mod = importasmod('\n            class Err(Exception):\n                pass\n            def fail():\n                return 0 / 0\n            def reraise():\n                try:\n                    fail()\n                except ZeroDivisionError as e:\n                    raise Err() from e\n            def unreraise():\n                try:\n                    reraise()\n                except Err as e:\n                    raise e.__cause__\n        ')
        excinfo = pytest.raises(ZeroDivisionError, mod.unreraise)
        r = excinfo.getrepr(style='short')
        r.toterminal(tw_mock)
        out = '\n'.join((line for line in tw_mock.lines if isinstance(line, str)))
        expected_out = textwrap.dedent('            :13: in unreraise\n                reraise()\n            :10: in reraise\n                raise Err() from e\n            E   test_exc_chain_repr_cycle0.mod.Err\n\n            During handling of the above exception, another exception occurred:\n            :15: in unreraise\n                raise e.__cause__\n            :8: in reraise\n                fail()\n            :5: in fail\n                return 0 / 0\n            E   ZeroDivisionError: division by zero')
        assert out == expected_out

    def test_exec_type_error_filter(self, importasmod):
        if False:
            i = 10
            return i + 15
        'See #7742'
        mod = importasmod('            def f():\n                exec("a = 1", {}, [])\n            ')
        with pytest.raises(TypeError) as excinfo:
            mod.f()
        excinfo.traceback.filter(excinfo)

@pytest.mark.parametrize('style', ['short', 'long'])
@pytest.mark.parametrize('encoding', [None, 'utf8', 'utf16'])
def test_repr_traceback_with_unicode(style, encoding):
    if False:
        print('Hello World!')
    if encoding is None:
        msg: str | bytes = '☹'
    else:
        msg = '☹'.encode(encoding)
    try:
        raise RuntimeError(msg)
    except RuntimeError:
        e_info = ExceptionInfo.from_current()
    formatter = FormattedExcinfo(style=style)
    repr_traceback = formatter.repr_traceback(e_info)
    assert repr_traceback is not None

def test_cwd_deleted(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        import os\n\n        def test(tmp_path):\n            os.chdir(tmp_path)\n            tmp_path.unlink()\n            assert False\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['* 1 failed in *'])
    result.stdout.no_fnmatch_line('*INTERNALERROR*')
    result.stderr.no_fnmatch_line('*INTERNALERROR*')

def test_regression_nagative_line_index(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    '\n    With Python 3.10 alphas, there was an INTERNALERROR reported in\n    https://github.com/pytest-dev/pytest/pull/8227\n    This test ensures it does not regress.\n    '
    pytester.makepyfile('\n        import ast\n        import pytest\n\n\n        def test_literal_eval():\n            with pytest.raises(ValueError, match="^$"):\n                ast.literal_eval("pytest")\n    ')
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(['* 1 failed in *'])
    result.stdout.no_fnmatch_line('*INTERNALERROR*')
    result.stderr.no_fnmatch_line('*INTERNALERROR*')

@pytest.mark.usefixtures('limited_recursion_depth')
def test_exception_repr_extraction_error_on_recursion():
    if False:
        print('Hello World!')
    '\n    Ensure we can properly detect a recursion error even\n    if some locals raise error on comparison (#2459).\n    '

    class numpy_like:

        def __eq__(self, other):
            if False:
                while True:
                    i = 10
            if type(other) is numpy_like:
                raise ValueError('The truth value of an array with more than one element is ambiguous.')

    def a(x):
        if False:
            print('Hello World!')
        return b(numpy_like())

    def b(x):
        if False:
            i = 10
            return i + 15
        return a(numpy_like())
    with pytest.raises(RuntimeError) as excinfo:
        a(numpy_like())
    matcher = LineMatcher(str(excinfo.getrepr()).splitlines())
    matcher.fnmatch_lines(['!!! Recursion error detected, but an error occurred locating the origin of recursion.', '*The following exception happened*', '*ValueError: The truth value of an array*'])

@pytest.mark.usefixtures('limited_recursion_depth')
def test_no_recursion_index_on_recursion_error():
    if False:
        return 10
    "\n    Ensure that we don't break in case we can't find the recursion index\n    during a recursion error (#2486).\n    "

    class RecursionDepthError:

        def __getattr__(self, attr):
            if False:
                while True:
                    i = 10
            return getattr(self, '_' + attr)
    with pytest.raises(RuntimeError) as excinfo:
        RecursionDepthError().trigger
    assert 'maximum recursion' in str(excinfo.getrepr())

def _exceptiongroup_common(pytester: Pytester, outer_chain: str, inner_chain: str, native: bool) -> None:
    if False:
        while True:
            i = 10
    pre_raise = 'exceptiongroup.' if not native else ''
    pre_catch = pre_raise if sys.version_info < (3, 11) else ''
    filestr = f'''\n    {('import exceptiongroup' if not native else '')}\n    import pytest\n\n    def f(): raise ValueError("From f()")\n    def g(): raise BaseException("From g()")\n\n    def inner(inner_chain):\n        excs = []\n        for callback in [f, g]:\n            try:\n                callback()\n            except BaseException as err:\n                excs.append(err)\n        if excs:\n            if inner_chain == "none":\n                raise {pre_raise}BaseExceptionGroup("Oops", excs)\n            try:\n                raise SyntaxError()\n            except SyntaxError as e:\n                if inner_chain == "from":\n                    raise {pre_raise}BaseExceptionGroup("Oops", excs) from e\n                else:\n                    raise {pre_raise}BaseExceptionGroup("Oops", excs)\n\n    def outer(outer_chain, inner_chain):\n        try:\n            inner(inner_chain)\n        except {pre_catch}BaseExceptionGroup as e:\n            if outer_chain == "none":\n                raise\n            if outer_chain == "from":\n                raise IndexError() from e\n            else:\n                raise IndexError()\n\n\n    def test():\n        outer("{outer_chain}", "{inner_chain}")\n    '''
    pytester.makepyfile(test_excgroup=filestr)
    result = pytester.runpytest()
    match_lines = []
    if inner_chain in ('another', 'from'):
        match_lines.append('SyntaxError: <no detail available>')
    match_lines += ['  + Exception Group Traceback (most recent call last):', f'  \\| {pre_catch}BaseExceptionGroup: Oops \\(2 sub-exceptions\\)', '    \\| ValueError: From f\\(\\)', '    \\| BaseException: From g\\(\\)', '=* short test summary info =*']
    if outer_chain in ('another', 'from'):
        match_lines.append('FAILED test_excgroup.py::test - IndexError')
    else:
        match_lines.append(f'FAILED test_excgroup.py::test - {pre_catch}BaseExceptionGroup: Oops \\(2.*')
    result.stdout.re_match_lines(match_lines)

@pytest.mark.skipif(sys.version_info < (3, 11), reason='Native ExceptionGroup not implemented')
@pytest.mark.parametrize('outer_chain', ['none', 'from', 'another'])
@pytest.mark.parametrize('inner_chain', ['none', 'from', 'another'])
def test_native_exceptiongroup(pytester: Pytester, outer_chain, inner_chain) -> None:
    if False:
        print('Hello World!')
    _exceptiongroup_common(pytester, outer_chain, inner_chain, native=True)

@pytest.mark.parametrize('outer_chain', ['none', 'from', 'another'])
@pytest.mark.parametrize('inner_chain', ['none', 'from', 'another'])
def test_exceptiongroup(pytester: Pytester, outer_chain, inner_chain) -> None:
    if False:
        while True:
            i = 10
    pytest.importorskip('exceptiongroup')
    _exceptiongroup_common(pytester, outer_chain, inner_chain, native=False)

@pytest.mark.parametrize('tbstyle', ('long', 'short', 'auto', 'line', 'native'))
def test_all_entries_hidden(pytester: Pytester, tbstyle: str) -> None:
    if False:
        print('Hello World!')
    'Regression test for #10903.'
    pytester.makepyfile('\n        def test():\n            __tracebackhide__ = True\n            1 / 0\n    ')
    result = pytester.runpytest('--tb', tbstyle)
    assert result.ret == 1
    if tbstyle != 'line':
        result.stdout.fnmatch_lines(['*ZeroDivisionError: division by zero'])
    if tbstyle not in ('line', 'native'):
        result.stdout.fnmatch_lines(['All traceback entries are hidden.*'])

def test_hidden_entries_of_chained_exceptions_are_not_shown(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Hidden entries of chained exceptions are not shown (#1904).'
    p = pytester.makepyfile('\n        def g1():\n            __tracebackhide__ = True\n            str.does_not_exist\n\n        def f3():\n            __tracebackhide__ = True\n            1 / 0\n\n        def f2():\n            try:\n                f3()\n            except Exception:\n                g1()\n\n        def f1():\n            __tracebackhide__ = True\n            f2()\n\n        def test():\n            f1()\n        ')
    result = pytester.runpytest(str(p), '--tb=short')
    assert result.ret == 1
    result.stdout.fnmatch_lines(['*.py:11: in f2', '    f3()', 'E   ZeroDivisionError: division by zero', '', 'During handling of the above exception, another exception occurred:', '*.py:20: in test', '    f1()', '*.py:13: in f2', '    g1()', "E   AttributeError:*'does_not_exist'"], consecutive=True)

def add_note(err: BaseException, msg: str) -> None:
    if False:
        print('Hello World!')
    'Adds a note to an exception inplace.'
    if sys.version_info < (3, 11):
        err.__notes__ = getattr(err, '__notes__', []) + [msg]
    else:
        err.add_note(msg)

@pytest.mark.parametrize('error,notes,match', [(Exception('test'), [], 'test'), (AssertionError('foo'), ['bar'], 'bar'), (AssertionError('foo'), ['bar', 'baz'], 'bar'), (AssertionError('foo'), ['bar', 'baz'], 'baz'), (ValueError('foo'), ['bar', 'baz'], re.compile('bar\\nbaz', re.MULTILINE)), (ValueError('foo'), ['bar', 'baz'], re.compile('BAZ', re.IGNORECASE))])
def test_check_error_notes_success(error: Exception, notes: list[str], match: str) -> None:
    if False:
        i = 10
        return i + 15
    for note in notes:
        add_note(error, note)
    with pytest.raises(Exception, match=match):
        raise error

@pytest.mark.parametrize('error, notes, match', [(Exception('test'), [], 'foo'), (AssertionError('foo'), ['bar'], 'baz'), (AssertionError('foo'), ['bar'], 'foo\nbaz')])
def test_check_error_notes_failure(error: Exception, notes: list[str], match: str) -> None:
    if False:
        i = 10
        return i + 15
    for note in notes:
        add_note(error, note)
    with pytest.raises(AssertionError):
        with pytest.raises(type(error), match=match):
            raise error