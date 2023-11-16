import dataclasses
import itertools
import re
import sys
import textwrap
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import hypothesis
from hypothesis import strategies
import pytest
from _pytest import fixtures
from _pytest import python
from _pytest.compat import getfuncargnames
from _pytest.compat import NOTSET
from _pytest.outcomes import fail
from _pytest.pytester import Pytester
from _pytest.python import Function
from _pytest.python import IdMaker
from _pytest.scope import Scope

class TestMetafunc:

    def Metafunc(self, func, config=None) -> python.Metafunc:
        if False:
            return 10

        class FuncFixtureInfoMock:
            name2fixturedefs: Dict[str, List[fixtures.FixtureDef[object]]] = {}

            def __init__(self, names):
                if False:
                    for i in range(10):
                        print('nop')
                self.names_closure = names

        @dataclasses.dataclass
        class FixtureManagerMock:
            config: Any

        @dataclasses.dataclass
        class SessionMock:
            _fixturemanager: FixtureManagerMock

        @dataclasses.dataclass
        class DefinitionMock(python.FunctionDefinition):
            _nodeid: str
            obj: object
        names = getfuncargnames(func)
        fixtureinfo: Any = FuncFixtureInfoMock(names)
        definition: Any = DefinitionMock._create(obj=func, _nodeid='mock::nodeid')
        definition._fixtureinfo = fixtureinfo
        definition.session = SessionMock(FixtureManagerMock({}))
        return python.Metafunc(definition, fixtureinfo, config, _ispytest=True)

    def test_no_funcargs(self) -> None:
        if False:
            i = 10
            return i + 15

        def function():
            if False:
                i = 10
                return i + 15
            pass
        metafunc = self.Metafunc(function)
        assert not metafunc.fixturenames
        repr(metafunc._calls)

    def test_function_basic(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def func(arg1, arg2='qwe'):
            if False:
                return 10
            pass
        metafunc = self.Metafunc(func)
        assert len(metafunc.fixturenames) == 1
        assert 'arg1' in metafunc.fixturenames
        assert metafunc.function is func
        assert metafunc.cls is None

    def test_parametrize_error(self) -> None:
        if False:
            return 10

        def func(x, y):
            if False:
                while True:
                    i = 10
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2])
        pytest.raises(ValueError, lambda : metafunc.parametrize('x', [5, 6]))
        pytest.raises(ValueError, lambda : metafunc.parametrize('x', [5, 6]))
        metafunc.parametrize('y', [1, 2])
        pytest.raises(ValueError, lambda : metafunc.parametrize('y', [5, 6]))
        pytest.raises(ValueError, lambda : metafunc.parametrize('y', [5, 6]))
        with pytest.raises(TypeError, match='^ids must be a callable or an iterable$'):
            metafunc.parametrize('y', [5, 6], ids=42)

    def test_parametrize_error_iterator(self) -> None:
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            raise NotImplementedError()

        class Exc(Exception):

            def __repr__(self):
                if False:
                    print('Hello World!')
                return 'Exc(from_gen)'

        def gen() -> Iterator[Union[int, None, Exc]]:
            if False:
                i = 10
                return i + 15
            yield 0
            yield None
            yield Exc()
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2], ids=gen())
        assert [(x.params, x.id) for x in metafunc._calls] == [({'x': 1}, '0'), ({'x': 2}, '2')]
        with pytest.raises(fail.Exception, match="In func: ids contains unsupported value Exc\\(from_gen\\) \\(type: <class .*Exc'>\\) at index 2. Supported types are: .*"):
            metafunc.parametrize('x', [1, 2, 3], ids=gen())

    def test_parametrize_bad_scope(self) -> None:
        if False:
            while True:
                i = 10

        def func(x):
            if False:
                print('Hello World!')
            pass
        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception, match="parametrize\\(\\) call in func got an unexpected scope value 'doggy'"):
            metafunc.parametrize('x', [1], scope='doggy')

    def test_parametrize_request_name(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        "Show proper error  when 'request' is used as a parameter name in parametrize (#6183)"

        def func(request):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError()
        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception, match="'request' is a reserved name and cannot be used in @pytest.mark.parametrize"):
            metafunc.parametrize('request', [1])

    def test_find_parametrized_scope(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Unit test for _find_parametrized_scope (#3941).'
        from _pytest.python import _find_parametrized_scope

        @dataclasses.dataclass
        class DummyFixtureDef:
            _scope: Scope
        fixtures_defs = cast(Dict[str, Sequence[fixtures.FixtureDef[object]]], dict(session_fix=[DummyFixtureDef(Scope.Session)], package_fix=[DummyFixtureDef(Scope.Package)], module_fix=[DummyFixtureDef(Scope.Module)], class_fix=[DummyFixtureDef(Scope.Class)], func_fix=[DummyFixtureDef(Scope.Function)], mixed_fix=[DummyFixtureDef(Scope.Module), DummyFixtureDef(Scope.Class)]))

        def find_scope(argnames, indirect):
            if False:
                print('Hello World!')
            return _find_parametrized_scope(argnames, fixtures_defs, indirect=indirect)
        assert find_scope(['func_fix'], indirect=True) == Scope.Function
        assert find_scope(['class_fix'], indirect=True) == Scope.Class
        assert find_scope(['module_fix'], indirect=True) == Scope.Module
        assert find_scope(['package_fix'], indirect=True) == Scope.Package
        assert find_scope(['session_fix'], indirect=True) == Scope.Session
        assert find_scope(['class_fix', 'func_fix'], indirect=True) == Scope.Function
        assert find_scope(['func_fix', 'session_fix'], indirect=True) == Scope.Function
        assert find_scope(['session_fix', 'class_fix'], indirect=True) == Scope.Class
        assert find_scope(['package_fix', 'session_fix'], indirect=True) == Scope.Package
        assert find_scope(['module_fix', 'session_fix'], indirect=True) == Scope.Module
        assert find_scope(['session_fix', 'module_fix'], indirect=False) == Scope.Function
        assert find_scope(['session_fix', 'module_fix'], indirect=['module_fix']) == Scope.Function
        assert find_scope(['session_fix', 'module_fix'], indirect=['session_fix', 'module_fix']) == Scope.Module
        assert find_scope(['mixed_fix'], indirect=True) == Scope.Class

    def test_parametrize_and_id(self) -> None:
        if False:
            while True:
                i = 10

        def func(x, y):
            if False:
                return 10
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2], ids=['basic', 'advanced'])
        metafunc.parametrize('y', ['abc', 'def'])
        ids = [x.id for x in metafunc._calls]
        assert ids == ['basic-abc', 'basic-def', 'advanced-abc', 'advanced-def']

    def test_parametrize_and_id_unicode(self) -> None:
        if False:
            print('Hello World!')
        'Allow unicode strings for "ids" parameter in Python 2 (##1905)'

        def func(x):
            if False:
                i = 10
                return i + 15
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2], ids=['basic', 'advanced'])
        ids = [x.id for x in metafunc._calls]
        assert ids == ['basic', 'advanced']

    def test_parametrize_with_wrong_number_of_ids(self) -> None:
        if False:
            return 10

        def func(x, y):
            if False:
                print('Hello World!')
            pass
        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception):
            metafunc.parametrize('x', [1, 2], ids=['basic'])
        with pytest.raises(fail.Exception):
            metafunc.parametrize(('x', 'y'), [('abc', 'def'), ('ghi', 'jkl')], ids=['one'])

    def test_parametrize_ids_iterator_without_mark(self) -> None:
        if False:
            i = 10
            return i + 15

        def func(x, y):
            if False:
                while True:
                    i = 10
            pass
        it = itertools.count()
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2], ids=it)
        metafunc.parametrize('y', [3, 4], ids=it)
        ids = [x.id for x in metafunc._calls]
        assert ids == ['0-2', '0-3', '1-2', '1-3']
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1, 2], ids=it)
        metafunc.parametrize('y', [3, 4], ids=it)
        ids = [x.id for x in metafunc._calls]
        assert ids == ['4-6', '4-7', '5-6', '5-7']

    def test_parametrize_empty_list(self) -> None:
        if False:
            return 10
        '#510'

        def func(y):
            if False:
                i = 10
                return i + 15
            pass

        class MockConfig:

            def getini(self, name):
                if False:
                    print('Hello World!')
                return ''

            @property
            def hook(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def pytest_make_parametrize_id(self, **kw):
                if False:
                    print('Hello World!')
                pass
        metafunc = self.Metafunc(func, MockConfig())
        metafunc.parametrize('y', [])
        assert 'skip' == metafunc._calls[0].marks[0].name

    def test_parametrize_with_userobjects(self) -> None:
        if False:
            return 10

        def func(x, y):
            if False:
                return 10
            pass
        metafunc = self.Metafunc(func)

        class A:
            pass
        metafunc.parametrize('x', [A(), A()])
        metafunc.parametrize('y', list('ab'))
        assert metafunc._calls[0].id == 'x0-a'
        assert metafunc._calls[1].id == 'x0-b'
        assert metafunc._calls[2].id == 'x1-a'
        assert metafunc._calls[3].id == 'x1-b'

    @hypothesis.given(strategies.text() | strategies.binary())
    @hypothesis.settings(deadline=400.0)
    def test_idval_hypothesis(self, value) -> None:
        if False:
            while True:
                i = 10
        escaped = IdMaker([], [], None, None, None, None, None)._idval(value, 'a', 6)
        assert isinstance(escaped, str)
        escaped.encode('ascii')

    def test_unicode_idval(self) -> None:
        if False:
            i = 10
            return i + 15
        "Test that Unicode strings outside the ASCII character set get\n        escaped, using byte escapes if they're in that range or unicode\n        escapes if they're not.\n\n        "
        values = [('', ''), ('ascii', 'ascii'), ('ação', 'a\\xe7\\xe3o'), ('josé@blah.com', 'jos\\xe9@blah.com'), ('δοκ.ιμή@παράδειγμα.δοκιμή', '\\u03b4\\u03bf\\u03ba.\\u03b9\\u03bc\\u03ae@\\u03c0\\u03b1\\u03c1\\u03ac\\u03b4\\u03b5\\u03b9\\u03b3\\u03bc\\u03b1.\\u03b4\\u03bf\\u03ba\\u03b9\\u03bc\\u03ae')]
        for (val, expected) in values:
            assert IdMaker([], [], None, None, None, None, None)._idval(val, 'a', 6) == expected

    def test_unicode_idval_with_config(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Unit test for expected behavior to obtain ids with\n        disable_test_id_escaping_and_forfeit_all_rights_to_community_support\n        option (#5294).'

        class MockConfig:

            def __init__(self, config):
                if False:
                    print('Hello World!')
                self.config = config

            @property
            def hook(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def pytest_make_parametrize_id(self, **kw):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def getini(self, name):
                if False:
                    i = 10
                    return i + 15
                return self.config[name]
        option = 'disable_test_id_escaping_and_forfeit_all_rights_to_community_support'
        values: List[Tuple[str, Any, str]] = [('ação', MockConfig({option: True}), 'ação'), ('ação', MockConfig({option: False}), 'a\\xe7\\xe3o')]
        for (val, config, expected) in values:
            actual = IdMaker([], [], None, None, config, None, None)._idval(val, 'a', 6)
            assert actual == expected

    def test_bytes_idval(self) -> None:
        if False:
            while True:
                i = 10
        'Unit test for the expected behavior to obtain ids for parametrized\n        bytes values: bytes objects are always escaped using "binary escape".'
        values = [(b'', ''), (b'\xc3\xb4\xff\xe4', '\\xc3\\xb4\\xff\\xe4'), (b'ascii', 'ascii'), ('αρά'.encode(), '\\xce\\xb1\\xcf\\x81\\xce\\xac')]
        for (val, expected) in values:
            assert IdMaker([], [], None, None, None, None, None)._idval(val, 'a', 6) == expected

    def test_class_or_function_idval(self) -> None:
        if False:
            print('Hello World!')
        'Unit test for the expected behavior to obtain ids for parametrized\n        values that are classes or functions: their __name__.'

        class TestClass:
            pass

        def test_function():
            if False:
                print('Hello World!')
            pass
        values = [(TestClass, 'TestClass'), (test_function, 'test_function')]
        for (val, expected) in values:
            assert IdMaker([], [], None, None, None, None, None)._idval(val, 'a', 6) == expected

    def test_notset_idval(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that a NOTSET value (used by an empty parameterset) generates\n        a proper ID.\n\n        Regression test for #7686.\n        '
        assert IdMaker([], [], None, None, None, None, None)._idval(NOTSET, 'a', 0) == 'a0'

    def test_idmaker_autoname(self) -> None:
        if False:
            i = 10
            return i + 15
        '#250'
        result = IdMaker(('a', 'b'), [pytest.param('string', 1.0), pytest.param('st-ring', 2.0)], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['string-1.0', 'st-ring-2.0']
        result = IdMaker(('a', 'b'), [pytest.param(object(), 1.0), pytest.param(object(), object())], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['a0-1.0', 'a1-b1']
        result = IdMaker(('a', 'b'), [pytest.param({}, b'\xc3\xb4')], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['a0-\\xc3\\xb4']

    def test_idmaker_with_bytes_regex(self) -> None:
        if False:
            i = 10
            return i + 15
        result = IdMaker('a', [pytest.param(re.compile(b'foo'), 1.0)], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['foo']

    def test_idmaker_native_strings(self) -> None:
        if False:
            print('Hello World!')
        result = IdMaker(('a', 'b'), [pytest.param(1.0, -1.1), pytest.param(2, -202), pytest.param('three', 'three hundred'), pytest.param(True, False), pytest.param(None, None), pytest.param(re.compile('foo'), re.compile('bar')), pytest.param(str, int), pytest.param(list('six'), [66, 66]), pytest.param({7}, set('seven')), pytest.param(tuple('eight'), (8, -8, 8)), pytest.param(b'\xc3\xb4', b'name'), pytest.param(b'\xc3\xb4', 'other'), pytest.param(1j, -2j)], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['1.0--1.1', '2--202', 'three-three hundred', 'True-False', 'None-None', 'foo-bar', 'str-int', 'a7-b7', 'a8-b8', 'a9-b9', '\\xc3\\xb4-name', '\\xc3\\xb4-other', '1j-(-0-2j)']

    def test_idmaker_non_printable_characters(self) -> None:
        if False:
            print('Hello World!')
        result = IdMaker(('s', 'n'), [pytest.param('\x00', 1), pytest.param('\x05', 2), pytest.param(b'\x00', 3), pytest.param(b'\x05', 4), pytest.param('\t', 5), pytest.param(b'\t', 6)], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['\\x00-1', '\\x05-2', '\\x00-3', '\\x05-4', '\\t-5', '\\t-6']

    def test_idmaker_manual_ids_must_be_printable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        result = IdMaker(('s',), [pytest.param('x00', id='hello \x00'), pytest.param('x05', id='hello \x05')], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['hello \\x00', 'hello \\x05']

    def test_idmaker_enum(self) -> None:
        if False:
            while True:
                i = 10
        enum = pytest.importorskip('enum')
        e = enum.Enum('Foo', 'one, two')
        result = IdMaker(('a', 'b'), [pytest.param(e.one, e.two)], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['Foo.one-Foo.two']

    def test_idmaker_idfn(self) -> None:
        if False:
            return 10
        '#351'

        def ids(val: object) -> Optional[str]:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(val, Exception):
                return repr(val)
            return None
        result = IdMaker(('a', 'b'), [pytest.param(10.0, IndexError()), pytest.param(20, KeyError()), pytest.param('three', [1, 2, 3])], ids, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['10.0-IndexError()', '20-KeyError()', 'three-b2']

    def test_idmaker_idfn_unique_names(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '#351'

        def ids(val: object) -> str:
            if False:
                i = 10
                return i + 15
            return 'a'
        result = IdMaker(('a', 'b'), [pytest.param(10.0, IndexError()), pytest.param(20, KeyError()), pytest.param('three', [1, 2, 3])], ids, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['a-a0', 'a-a1', 'a-a2']

    def test_idmaker_with_idfn_and_config(self) -> None:
        if False:
            while True:
                i = 10
        'Unit test for expected behavior to create ids with idfn and\n        disable_test_id_escaping_and_forfeit_all_rights_to_community_support\n        option (#5294).\n        '

        class MockConfig:

            def __init__(self, config):
                if False:
                    return 10
                self.config = config

            @property
            def hook(self):
                if False:
                    while True:
                        i = 10
                return self

            def pytest_make_parametrize_id(self, **kw):
                if False:
                    print('Hello World!')
                pass

            def getini(self, name):
                if False:
                    return 10
                return self.config[name]
        option = 'disable_test_id_escaping_and_forfeit_all_rights_to_community_support'
        values: List[Tuple[Any, str]] = [(MockConfig({option: True}), 'ação'), (MockConfig({option: False}), 'a\\xe7\\xe3o')]
        for (config, expected) in values:
            result = IdMaker(('a',), [pytest.param('string')], lambda _: 'ação', None, config, None, None).make_unique_parameterset_ids()
            assert result == [expected]

    def test_idmaker_with_ids_and_config(self) -> None:
        if False:
            return 10
        'Unit test for expected behavior to create ids with ids and\n        disable_test_id_escaping_and_forfeit_all_rights_to_community_support\n        option (#5294).\n        '

        class MockConfig:

            def __init__(self, config):
                if False:
                    print('Hello World!')
                self.config = config

            @property
            def hook(self):
                if False:
                    print('Hello World!')
                return self

            def pytest_make_parametrize_id(self, **kw):
                if False:
                    i = 10
                    return i + 15
                pass

            def getini(self, name):
                if False:
                    return 10
                return self.config[name]
        option = 'disable_test_id_escaping_and_forfeit_all_rights_to_community_support'
        values: List[Tuple[Any, str]] = [(MockConfig({option: True}), 'ação'), (MockConfig({option: False}), 'a\\xe7\\xe3o')]
        for (config, expected) in values:
            result = IdMaker(('a',), [pytest.param('string')], None, ['ação'], config, None, None).make_unique_parameterset_ids()
            assert result == [expected]

    def test_idmaker_duplicated_empty_str(self) -> None:
        if False:
            print('Hello World!')
        'Regression test for empty strings parametrized more than once (#11563).'
        result = IdMaker(('a',), [pytest.param(''), pytest.param('')], None, None, None, None, None).make_unique_parameterset_ids()
        assert result == ['0', '1']

    def test_parametrize_ids_exception(self, pytester: Pytester) -> None:
        if False:
            return 10
        '\n        :param pytester: the instance of Pytester class, a temporary\n        test directory.\n        '
        pytester.makepyfile('\n                import pytest\n\n                def ids(arg):\n                    raise Exception("bad ids")\n\n                @pytest.mark.parametrize("arg", ["a", "b"], ids=ids)\n                def test_foo(arg):\n                    pass\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*Exception: bad ids', "*test_foo: error raised while trying to determine id of parameter 'arg' at position 0"])

    def test_parametrize_ids_returns_non_string(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('            import pytest\n\n            def ids(d):\n                return d\n\n            @pytest.mark.parametrize("arg", ({1: 2}, {3, 4}), ids=ids)\n            def test(arg):\n                assert arg\n\n            @pytest.mark.parametrize("arg", (1, 2.0, True), ids=ids)\n            def test_int(arg):\n                assert arg\n            ')
        result = pytester.runpytest('-vv', '-s')
        result.stdout.fnmatch_lines(['test_parametrize_ids_returns_non_string.py::test[arg0] PASSED', 'test_parametrize_ids_returns_non_string.py::test[arg1] PASSED', 'test_parametrize_ids_returns_non_string.py::test_int[1] PASSED', 'test_parametrize_ids_returns_non_string.py::test_int[2.0] PASSED', 'test_parametrize_ids_returns_non_string.py::test_int[True] PASSED'])

    def test_idmaker_with_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        result = IdMaker(('a', 'b'), [pytest.param(1, 2), pytest.param(3, 4)], None, ['a', None], None, None, None).make_unique_parameterset_ids()
        assert result == ['a', '3-4']

    def test_idmaker_with_paramset_id(self) -> None:
        if False:
            i = 10
            return i + 15
        result = IdMaker(('a', 'b'), [pytest.param(1, 2, id='me'), pytest.param(3, 4, id='you')], None, ['a', None], None, None, None).make_unique_parameterset_ids()
        assert result == ['me', 'you']

    def test_idmaker_with_ids_unique_names(self) -> None:
        if False:
            i = 10
            return i + 15
        result = IdMaker('a', list(map(pytest.param, [1, 2, 3, 4, 5])), None, ['a', 'a', 'b', 'c', 'b'], None, None, None).make_unique_parameterset_ids()
        assert result == ['a0', 'a1', 'b0', 'c', 'b1']

    def test_parametrize_indirect(self) -> None:
        if False:
            while True:
                i = 10
        '#714'

        def func(x, y):
            if False:
                while True:
                    i = 10
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x', [1], indirect=True)
        metafunc.parametrize('y', [2, 3], indirect=True)
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1, y=2)
        assert metafunc._calls[1].params == dict(x=1, y=3)

    def test_parametrize_indirect_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '#714'

        def func(x, y):
            if False:
                i = 10
                return i + 15
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x, y', [('a', 'b')], indirect=['x'])
        assert metafunc._calls[0].params == dict(x='a', y='b')
        assert list(metafunc._arg2fixturedefs.keys()) == ['y']

    def test_parametrize_indirect_list_all(self) -> None:
        if False:
            while True:
                i = 10
        '#714'

        def func(x, y):
            if False:
                i = 10
                return i + 15
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x, y', [('a', 'b')], indirect=['x', 'y'])
        assert metafunc._calls[0].params == dict(x='a', y='b')
        assert list(metafunc._arg2fixturedefs.keys()) == []

    def test_parametrize_indirect_list_empty(self) -> None:
        if False:
            return 10
        '#714'

        def func(x, y):
            if False:
                i = 10
                return i + 15
            pass
        metafunc = self.Metafunc(func)
        metafunc.parametrize('x, y', [('a', 'b')], indirect=[])
        assert metafunc._calls[0].params == dict(x='a', y='b')
        assert list(metafunc._arg2fixturedefs.keys()) == ['x', 'y']

    def test_parametrize_indirect_wrong_type(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def func(x, y):
            if False:
                while True:
                    i = 10
            pass
        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception, match='In func: expected Sequence or boolean for indirect, got dict'):
            metafunc.parametrize('x, y', [('a', 'b')], indirect={})

    def test_parametrize_indirect_list_functional(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        "\n        #714\n        Test parametrization with 'indirect' parameter applied on\n        particular arguments. As y is direct, its value should\n        be used directly rather than being passed to the fixture y.\n\n        :param pytester: the instance of Pytester class, a temporary\n        test directory.\n        "
        pytester.makepyfile("\n            import pytest\n            @pytest.fixture(scope='function')\n            def x(request):\n                return request.param * 3\n            @pytest.fixture(scope='function')\n            def y(request):\n                return request.param * 2\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect=['x'])\n            def test_simple(x,y):\n                assert len(x) == 3\n                assert len(y) == 1\n        ")
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_simple*a-b*', '*1 passed*'])

    def test_parametrize_indirect_list_error(self) -> None:
        if False:
            return 10
        '#714'

        def func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            pass
        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception):
            metafunc.parametrize('x, y', [('a', 'b')], indirect=['x', 'z'])

    def test_parametrize_uses_no_fixture_error_indirect_false(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        "The 'uses no fixture' error tells the user at collection time\n        that the parametrize data they've set up doesn't correspond to the\n        fixtures in their test function, rather than silently ignoring this\n        and letting the test potentially pass.\n\n        #714\n        "
        pytester.makepyfile("\n            import pytest\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect=False)\n            def test_simple(x):\n                assert len(x) == 3\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*uses no argument 'y'*"])

    def test_parametrize_uses_no_fixture_error_indirect_true(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        '#714'
        pytester.makepyfile("\n            import pytest\n            @pytest.fixture(scope='function')\n            def x(request):\n                return request.param * 3\n            @pytest.fixture(scope='function')\n            def y(request):\n                return request.param * 2\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect=True)\n            def test_simple(x):\n                assert len(x) == 3\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*uses no fixture 'y'*"])

    def test_parametrize_indirect_uses_no_fixture_error_indirect_string(self, pytester: Pytester) -> None:
        if False:
            return 10
        '#714'
        pytester.makepyfile("\n            import pytest\n            @pytest.fixture(scope='function')\n            def x(request):\n                return request.param * 3\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect='y')\n            def test_simple(x):\n                assert len(x) == 3\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*uses no fixture 'y'*"])

    def test_parametrize_indirect_uses_no_fixture_error_indirect_list(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        '#714'
        pytester.makepyfile("\n            import pytest\n            @pytest.fixture(scope='function')\n            def x(request):\n                return request.param * 3\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect=['y'])\n            def test_simple(x):\n                assert len(x) == 3\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*uses no fixture 'y'*"])

    def test_parametrize_argument_not_in_indirect_list(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        '#714'
        pytester.makepyfile("\n            import pytest\n            @pytest.fixture(scope='function')\n            def x(request):\n                return request.param * 3\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')], indirect=['x'])\n            def test_simple(x):\n                assert len(x) == 3\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*uses no argument 'y'*"])

    def test_parametrize_gives_indicative_error_on_function_with_default_argument(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n\n            @pytest.mark.parametrize('x, y', [('a', 'b')])\n            def test_simple(x, y=1):\n                assert len(x) == 1\n        ")
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(["*already takes an argument 'y' with a default value"])

    def test_parametrize_functional(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize('x', [1,2], indirect=True)\n                metafunc.parametrize('y', [2])\n            @pytest.fixture\n            def x(request):\n                return request.param * 10\n\n            def test_simple(x,y):\n                assert x in (10,20)\n                assert y == 2\n        ")
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_simple*1-2*', '*test_simple*2-2*', '*2 passed*'])

    def test_parametrize_onearg(self) -> None:
        if False:
            while True:
                i = 10
        metafunc = self.Metafunc(lambda x: None)
        metafunc.parametrize('x', [1, 2])
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1)
        assert metafunc._calls[0].id == '1'
        assert metafunc._calls[1].params == dict(x=2)
        assert metafunc._calls[1].id == '2'

    def test_parametrize_onearg_indirect(self) -> None:
        if False:
            i = 10
            return i + 15
        metafunc = self.Metafunc(lambda x: None)
        metafunc.parametrize('x', [1, 2], indirect=True)
        assert metafunc._calls[0].params == dict(x=1)
        assert metafunc._calls[0].id == '1'
        assert metafunc._calls[1].params == dict(x=2)
        assert metafunc._calls[1].id == '2'

    def test_parametrize_twoargs(self) -> None:
        if False:
            i = 10
            return i + 15
        metafunc = self.Metafunc(lambda x, y: None)
        metafunc.parametrize(('x', 'y'), [(1, 2), (3, 4)])
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1, y=2)
        assert metafunc._calls[0].id == '1-2'
        assert metafunc._calls[1].params == dict(x=3, y=4)
        assert metafunc._calls[1].id == '3-4'

    def test_high_scoped_parametrize_reordering(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("arg2", [3, 4])\n            @pytest.mark.parametrize("arg1", [0, 1, 2], scope=\'module\')\n            def test1(arg1, arg2):\n                pass\n\n            def test2():\n                pass\n\n            @pytest.mark.parametrize("arg1", [0, 1, 2], scope=\'module\')\n            def test3(arg1):\n                pass\n        ')
        result = pytester.runpytest('--collect-only')
        result.stdout.re_match_lines(['  <Function test1\\[0-3\\]>', '  <Function test1\\[0-4\\]>', '  <Function test3\\[0\\]>', '  <Function test1\\[1-3\\]>', '  <Function test1\\[1-4\\]>', '  <Function test3\\[1\\]>', '  <Function test1\\[2-3\\]>', '  <Function test1\\[2-4\\]>', '  <Function test3\\[2\\]>', '  <Function test2>'])

    def test_parametrize_multiple_times(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            pytestmark = pytest.mark.parametrize("x", [1,2])\n            def test_func(x):\n                assert 0, x\n            class TestClass(object):\n                pytestmark = pytest.mark.parametrize("y", [3,4])\n                def test_meth(self, x, y):\n                    assert 0, x\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.assert_outcomes(failed=6)

    def test_parametrize_CSV(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.parametrize("x, y,", [(1,2), (2,3)])\n            def test_func(x, y):\n                assert x+1 == y\n        ')
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_parametrize_class_scenarios(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n        # same as doc/en/example/parametrize scenario example\n        def pytest_generate_tests(metafunc):\n            idlist = []\n            argvalues = []\n            for scenario in metafunc.cls.scenarios:\n                idlist.append(scenario[0])\n                items = scenario[1].items()\n                argnames = [x[0] for x in items]\n                argvalues.append(([x[1] for x in items]))\n            metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")\n\n        class Test(object):\n               scenarios = [[\'1\', {\'arg\': {1: 2}, "arg2": "value2"}],\n                            [\'2\', {\'arg\':\'value2\', "arg2": "value2"}]]\n\n               def test_1(self, arg, arg2):\n                  pass\n\n               def test_2(self, arg2, arg):\n                  pass\n\n               def test_3(self, arg, arg2):\n                  pass\n        ')
        result = pytester.runpytest('-v')
        assert result.ret == 0
        result.stdout.fnmatch_lines('\n            *test_1*1*\n            *test_2*1*\n            *test_3*1*\n            *test_1*2*\n            *test_2*2*\n            *test_3*2*\n            *6 passed*\n        ')

class TestMetafuncFunctional:

    def test_attributes(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile("\n            # assumes that generate/provide runs in the same process\n            import sys, pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize('metafunc', [metafunc])\n\n            @pytest.fixture\n            def metafunc(request):\n                return request.param\n\n            def test_function(metafunc, pytestconfig):\n                assert metafunc.config == pytestconfig\n                assert metafunc.module.__name__ == __name__\n                assert metafunc.function == test_function\n                assert metafunc.cls is None\n\n            class TestClass(object):\n                def test_method(self, metafunc, pytestconfig):\n                    assert metafunc.config == pytestconfig\n                    assert metafunc.module.__name__ == __name__\n                    unbound = TestClass.test_method\n                    assert metafunc.function == unbound\n                    assert metafunc.cls == TestClass\n        ")
        result = pytester.runpytest(p, '-v')
        result.assert_outcomes(passed=2)

    def test_two_functions(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile("\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize('arg1', [10, 20], ids=['0', '1'])\n\n            def test_func1(arg1):\n                assert arg1 == 10\n\n            def test_func2(arg1):\n                assert arg1 in (10, 20)\n        ")
        result = pytester.runpytest('-v', p)
        result.stdout.fnmatch_lines(['*test_func1*0*PASS*', '*test_func1*1*FAIL*', '*test_func2*PASS*', '*test_func2*PASS*', '*1 failed, 3 passed*'])

    def test_noself_in_method(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.makepyfile("\n            def pytest_generate_tests(metafunc):\n                assert 'xyz' not in metafunc.fixturenames\n\n            class TestHello(object):\n                def test_hello(xyz):\n                    pass\n        ")
        result = pytester.runpytest(p)
        result.assert_outcomes(passed=1)

    def test_generate_tests_in_class(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p = pytester.makepyfile('\n            class TestClass(object):\n                def pytest_generate_tests(self, metafunc):\n                    metafunc.parametrize(\'hello\', [\'world\'], ids=[\'hellow\'])\n\n                def test_myfunc(self, hello):\n                    assert hello == "world"\n        ')
        result = pytester.runpytest('-v', p)
        result.stdout.fnmatch_lines(['*test_myfunc*hello*PASS*', '*1 passed*'])

    def test_two_functions_not_same_instance(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.makepyfile('\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(\'arg1\', [10, 20], ids=["0", "1"])\n\n            class TestClass(object):\n                def test_func(self, arg1):\n                    assert not hasattr(self, \'x\')\n                    self.x = 1\n        ')
        result = pytester.runpytest('-v', p)
        result.stdout.fnmatch_lines(['*test_func*0*PASS*', '*test_func*1*PASS*', '*2 pass*'])

    def test_issue28_setup_method_in_generate_tests(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile("\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize('arg1', [1])\n\n            class TestClass(object):\n                def test_method(self, arg1):\n                    assert arg1 == self.val\n                def setup_method(self, func):\n                    self.val = 1\n            ")
        result = pytester.runpytest(p)
        result.assert_outcomes(passed=1)

    def test_parametrize_functional2(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize("arg1", [1,2])\n                metafunc.parametrize("arg2", [4,5])\n            def test_hello(arg1, arg2):\n                assert 0, (arg1, arg2)\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*(1, 4)*', '*(1, 5)*', '*(2, 4)*', '*(2, 5)*', '*4 failed*'])

    def test_parametrize_and_inner_getfixturevalue(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.makepyfile('\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize("arg1", [1], indirect=True)\n                metafunc.parametrize("arg2", [10], indirect=True)\n\n            import pytest\n            @pytest.fixture\n            def arg1(request):\n                x = request.getfixturevalue("arg2")\n                return x + request.param\n\n            @pytest.fixture\n            def arg2(request):\n                return request.param\n\n            def test_func1(arg1, arg2):\n                assert arg1 == 11\n        ')
        result = pytester.runpytest('-v', p)
        result.stdout.fnmatch_lines(['*test_func1*1*PASS*', '*1 passed*'])

    def test_parametrize_on_setup_arg(self, pytester: Pytester) -> None:
        if False:
            return 10
        p = pytester.makepyfile('\n            def pytest_generate_tests(metafunc):\n                assert "arg1" in metafunc.fixturenames\n                metafunc.parametrize("arg1", [1], indirect=True)\n\n            import pytest\n            @pytest.fixture\n            def arg1(request):\n                return request.param\n\n            @pytest.fixture\n            def arg2(request, arg1):\n                return 10 * arg1\n\n            def test_func(arg2):\n                assert arg2 == 10\n        ')
        result = pytester.runpytest('-v', p)
        result.stdout.fnmatch_lines(['*test_func*1*PASS*', '*1 passed*'])

    def test_parametrize_with_ids(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeini('\n            [pytest]\n            console_output_style=classic\n        ')
        pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(("a", "b"), [(1,1), (1,2)],\n                                     ids=["basic", "advanced"])\n\n            def test_function(a, b):\n                assert a == b\n        ')
        result = pytester.runpytest('-v')
        assert result.ret == 1
        result.stdout.fnmatch_lines_random(['*test_function*basic*PASSED', '*test_function*advanced*FAILED'])

    def test_parametrize_without_ids(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(("a", "b"),\n                                     [(1,object()), (1.3,object())])\n\n            def test_function(a, b):\n                assert 1\n        ')
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines('\n            *test_function*1-b0*\n            *test_function*1.3-b1*\n        ')

    def test_parametrize_with_None_in_ids(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(("a", "b"), [(1,1), (1,1), (1,2)],\n                                     ids=["basic", None, "advanced"])\n\n            def test_function(a, b):\n                assert a == b\n        ')
        result = pytester.runpytest('-v')
        assert result.ret == 1
        result.stdout.fnmatch_lines_random(['*test_function*basic*PASSED*', '*test_function*1-1*PASSED*', '*test_function*advanced*FAILED*'])

    def test_fixture_parametrized_empty_ids(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Fixtures parametrized with empty ids cause an internal error (#1849).'
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture(scope="module", ids=[], params=[])\n            def temp(request):\n               return request.param\n\n            def test_temp(temp):\n                 pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 1 skipped *'])

    def test_parametrized_empty_ids(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Tests parametrized with empty ids cause an internal error (#1849).'
        pytester.makepyfile("\n            import pytest\n\n            @pytest.mark.parametrize('temp', [], ids=list())\n            def test_temp(temp):\n                 pass\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 1 skipped *'])

    def test_parametrized_ids_invalid_type(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Test error with non-strings/non-ints, without generator (#1857).'
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("x, expected", [(1, 2), (3, 4), (5, 6)], ids=(None, 2, OSError()))\n            def test_ids_numbers(x,expected):\n                assert x * 2 == expected\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["In test_ids_numbers: ids contains unsupported value OSError() (type: <class 'OSError'>) at index 2. Supported types are: str, bytes, int, float, complex, bool, enum, regex or anything with a __name__."])

    def test_parametrize_with_identical_ids_get_unique_names(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                metafunc.parametrize(("a", "b"), [(1,1), (1,2)],\n                                     ids=["a", "a"])\n\n            def test_function(a, b):\n                assert a == b\n        ')
        result = pytester.runpytest('-v')
        assert result.ret == 1
        result.stdout.fnmatch_lines_random(['*test_function*a0*PASSED*', '*test_function*a1*FAILED*'])

    @pytest.mark.parametrize(('scope', 'length'), [('module', 2), ('function', 4)])
    def test_parametrize_scope_overrides(self, pytester: Pytester, scope: str, length: int) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n            values = []\n            def pytest_generate_tests(metafunc):\n                if "arg" in metafunc.fixturenames:\n                    metafunc.parametrize("arg", [1,2], indirect=True,\n                                         scope=%r)\n            @pytest.fixture\n            def arg(request):\n                values.append(request.param)\n                return request.param\n            def test_hello(arg):\n                assert arg in (1,2)\n            def test_world(arg):\n                assert arg in (1,2)\n            def test_checklength():\n                assert len(values) == %d\n        ' % (scope, length))
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=5)

    def test_parametrize_issue323(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture(scope='module', params=range(966))\n            def foo(request):\n                return request.param\n\n            def test_it(foo):\n                pass\n            def test_it2(foo):\n                pass\n        ")
        reprec = pytester.inline_run('--collect-only')
        assert not reprec.getcalls('pytest_internalerror')

    def test_usefixtures_seen_in_generate_tests(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                assert "abc" in metafunc.fixturenames\n                metafunc.parametrize("abc", [1])\n\n            @pytest.mark.usefixtures("abc")\n            def test_function():\n                pass\n        ')
        reprec = pytester.runpytest()
        reprec.assert_outcomes(passed=1)

    def test_generate_tests_only_done_in_subdir(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        sub1 = pytester.mkpydir('sub1')
        sub2 = pytester.mkpydir('sub2')
        sub1.joinpath('conftest.py').write_text(textwrap.dedent('                def pytest_generate_tests(metafunc):\n                    assert metafunc.function.__name__ == "test_1"\n                '), encoding='utf-8')
        sub2.joinpath('conftest.py').write_text(textwrap.dedent('                def pytest_generate_tests(metafunc):\n                    assert metafunc.function.__name__ == "test_2"\n                '), encoding='utf-8')
        sub1.joinpath('test_in_sub1.py').write_text('def test_1(): pass', encoding='utf-8')
        sub2.joinpath('test_in_sub2.py').write_text('def test_2(): pass', encoding='utf-8')
        result = pytester.runpytest('--keep-duplicates', '-v', '-s', sub1, sub2, sub1)
        result.assert_outcomes(passed=3)

    def test_generate_same_function_names_issue403(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n\n            def make_tests():\n                @pytest.mark.parametrize("x", range(2))\n                def test_foo(x):\n                    pass\n                return test_foo\n\n            test_x = make_tests()\n            test_y = make_tests()\n        ')
        reprec = pytester.runpytest()
        reprec.assert_outcomes(passed=4)

    def test_parametrize_misspelling(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        '#463'
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrise("x", range(2))\n            def test_foo(x):\n                pass\n        ')
        result = pytester.runpytest('--collect-only')
        result.stdout.fnmatch_lines(['collected 0 items / 1 error', '', '*= ERRORS =*', '*_ ERROR collecting test_parametrize_misspelling.py _*', 'test_parametrize_misspelling.py:3: in <module>', '    @pytest.mark.parametrise("x", range(2))', "E   Failed: Unknown 'parametrise' mark, did you mean 'parametrize'?", '*! Interrupted: 1 error during collection !*', '*= no tests collected, 1 error in *'])

    @pytest.mark.parametrize('scope', ['class', 'package'])
    def test_parametrize_missing_scope_doesnt_crash(self, pytester: Pytester, scope: str) -> None:
        if False:
            while True:
                i = 10
        "Doesn't crash when parametrize(scope=<scope>) is used without a\n        corresponding <scope> node."
        pytester.makepyfile(f'\n            import pytest\n\n            @pytest.mark.parametrize("x", [0], scope="{scope}")\n            def test_it(x): pass\n            ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_parametrize_module_level_test_with_class_scope(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        '\n        Test that a class-scoped parametrization without a corresponding `Class`\n        gets module scope, i.e. we only create a single FixtureDef for it per module.\n        '
        module = pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("x", [0, 1], scope="class")\n            def test_1(x):\n                pass\n\n            @pytest.mark.parametrize("x", [1, 2], scope="module")\n            def test_2(x):\n                pass\n        ')
        (test_1_0, _, test_2_0, _) = pytester.genitems((pytester.getmodulecol(module),))
        assert isinstance(test_1_0, Function)
        assert test_1_0.name == 'test_1[0]'
        test_1_fixture_x = test_1_0._fixtureinfo.name2fixturedefs['x'][-1]
        assert isinstance(test_2_0, Function)
        assert test_2_0.name == 'test_2[1]'
        test_2_fixture_x = test_2_0._fixtureinfo.name2fixturedefs['x'][-1]
        assert test_1_fixture_x is test_2_fixture_x

    def test_reordering_with_scopeless_and_just_indirect_parametrization(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            import pytest\n\n            @pytest.fixture(scope="package")\n            def fixture1():\n                pass\n            ')
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture(scope="module")\n            def fixture0():\n                pass\n\n            @pytest.fixture(scope="module")\n            def fixture1(fixture0):\n                pass\n\n            @pytest.mark.parametrize("fixture1", [0], indirect=True)\n            def test_0(fixture1):\n                pass\n\n            @pytest.fixture(scope="module")\n            def fixture():\n                pass\n\n            @pytest.mark.parametrize("fixture", [0], indirect=True)\n            def test_1(fixture):\n                pass\n\n            def test_2():\n                pass\n\n            class Test:\n                @pytest.fixture(scope="class")\n                def fixture(self, fixture):\n                    pass\n\n                @pytest.mark.parametrize("fixture", [0], indirect=True)\n                def test_3(self, fixture):\n                    pass\n            ')
        result = pytester.runpytest('-v')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*test_0*', '*test_1*', '*test_2*', '*test_3*'])

class TestMetafuncFunctionalAuto:
    """Tests related to automatically find out the correct scope for
    parametrized tests (#1832)."""

    def test_parametrize_auto_scope(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture(scope=\'session\', autouse=True)\n            def fixture():\n                return 1\n\n            @pytest.mark.parametrize(\'animal\', ["dog", "cat"])\n            def test_1(animal):\n                assert animal in (\'dog\', \'cat\')\n\n            @pytest.mark.parametrize(\'animal\', [\'fish\'])\n            def test_2(animal):\n                assert animal == \'fish\'\n\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 3 passed *'])

    def test_parametrize_auto_scope_indirect(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture(scope=\'session\')\n            def echo(request):\n                return request.param\n\n            @pytest.mark.parametrize(\'animal, echo\', [("dog", 1), ("cat", 2)], indirect=[\'echo\'])\n            def test_1(animal, echo):\n                assert animal in (\'dog\', \'cat\')\n                assert echo in (1, 2, 3)\n\n            @pytest.mark.parametrize(\'animal, echo\', [(\'fish\', 3)], indirect=[\'echo\'])\n            def test_2(animal, echo):\n                assert animal == \'fish\'\n                assert echo in (1, 2, 3)\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 3 passed *'])

    def test_parametrize_auto_scope_override_fixture(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture(scope=\'session\', autouse=True)\n            def animal():\n                return \'fox\'\n\n            @pytest.mark.parametrize(\'animal\', ["dog", "cat"])\n            def test_1(animal):\n                assert animal in (\'dog\', \'cat\')\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 2 passed *'])

    def test_parametrize_all_indirects(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture()\n            def animal(request):\n                return request.param\n\n            @pytest.fixture(scope=\'session\')\n            def echo(request):\n                return request.param\n\n            @pytest.mark.parametrize(\'animal, echo\', [("dog", 1), ("cat", 2)], indirect=True)\n            def test_1(animal, echo):\n                assert animal in (\'dog\', \'cat\')\n                assert echo in (1, 2, 3)\n\n            @pytest.mark.parametrize(\'animal, echo\', [("fish", 3)], indirect=True)\n            def test_2(animal, echo):\n                assert animal == \'fish\'\n                assert echo in (1, 2, 3)\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 3 passed *'])

    def test_parametrize_some_arguments_auto_scope(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            while True:
                i = 10
        'Integration test for (#3941)'
        class_fix_setup: List[object] = []
        monkeypatch.setattr(sys, 'class_fix_setup', class_fix_setup, raising=False)
        func_fix_setup: List[object] = []
        monkeypatch.setattr(sys, 'func_fix_setup', func_fix_setup, raising=False)
        pytester.makepyfile("\n            import pytest\n            import sys\n\n            @pytest.fixture(scope='class', autouse=True)\n            def class_fix(request):\n                sys.class_fix_setup.append(request.param)\n\n            @pytest.fixture(autouse=True)\n            def func_fix():\n                sys.func_fix_setup.append(True)\n\n            @pytest.mark.parametrize('class_fix', [10, 20], indirect=True)\n            class Test:\n                def test_foo(self):\n                    pass\n                def test_bar(self):\n                    pass\n            ")
        result = pytester.runpytest_inprocess()
        result.stdout.fnmatch_lines(['* 4 passed in *'])
        assert func_fix_setup == [True] * 4
        assert class_fix_setup == [10, 20]

    def test_parametrize_issue634(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture(scope='module')\n            def foo(request):\n                print('preparing foo-%d' % request.param)\n                return 'foo-%d' % request.param\n\n            def test_one(foo):\n                pass\n\n            def test_two(foo):\n                pass\n\n            test_two.test_with = (2, 3)\n\n            def pytest_generate_tests(metafunc):\n                params = (1, 2, 3, 4)\n                if not 'foo' in metafunc.fixturenames:\n                    return\n\n                test_with = getattr(metafunc.function, 'test_with', None)\n                if test_with:\n                    params = test_with\n                metafunc.parametrize('foo', params, indirect=True)\n        ")
        result = pytester.runpytest('-s')
        output = result.stdout.str()
        assert output.count('preparing foo-2') == 1
        assert output.count('preparing foo-3') == 1

class TestMarkersWithParametrization:
    """#308"""

    def test_simple_mark(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        s = '\n            import pytest\n\n            @pytest.mark.foo\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(1, 3, marks=pytest.mark.bar),\n                (2, 3),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        items = pytester.getitems(s)
        assert len(items) == 3
        for item in items:
            assert 'foo' in item.keywords
        assert 'bar' not in items[0].keywords
        assert 'bar' in items[1].keywords
        assert 'bar' not in items[2].keywords

    def test_select_based_on_mark(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(2, 3, marks=pytest.mark.foo),\n                (3, 4),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        rec = pytester.inline_run('-m', 'foo')
        (passed, skipped, fail) = rec.listoutcomes()
        assert len(passed) == 1
        assert len(skipped) == 0
        assert len(fail) == 0

    def test_simple_xfail(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(1, 3, marks=pytest.mark.xfail),\n                (2, 3),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)

    def test_simple_xfail_single_argname(self, pytester: Pytester) -> None:
        if False:
            return 10
        s = '\n            import pytest\n\n            @pytest.mark.parametrize("n", [\n                2,\n                pytest.param(3, marks=pytest.mark.xfail),\n                4,\n            ])\n            def test_isEven(n):\n                assert n % 2 == 0\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)

    def test_xfail_with_arg(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(1, 3, marks=pytest.mark.xfail("True")),\n                (2, 3),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)

    def test_xfail_with_kwarg(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(1, 3, marks=pytest.mark.xfail(reason="some bug")),\n                (2, 3),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)

    def test_xfail_with_arg_and_kwarg(self, pytester: Pytester) -> None:
        if False:
            return 10
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(1, 3, marks=pytest.mark.xfail("True", reason="some bug")),\n                (2, 3),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=1)

    @pytest.mark.parametrize('strict', [True, False])
    def test_xfail_passing_is_xpass(self, pytester: Pytester, strict: bool) -> None:
        if False:
            return 10
        s = '\n            import pytest\n\n            m = pytest.mark.xfail("sys.version_info > (0, 0, 0)", reason="some bug", strict={strict})\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                (1, 2),\n                pytest.param(2, 3, marks=m),\n                (3, 4),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '.format(strict=strict)
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        (passed, failed) = (2, 1) if strict else (3, 0)
        reprec.assertoutcome(passed=passed, failed=failed)

    def test_parametrize_called_in_generate_tests(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        s = '\n            import pytest\n\n\n            def pytest_generate_tests(metafunc):\n                passingTestData = [(1, 2),\n                                   (2, 3)]\n                failingTestData = [(1, 3),\n                                   (2, 2)]\n\n                testData = passingTestData + [pytest.param(*d, marks=pytest.mark.xfail)\n                                  for d in failingTestData]\n                metafunc.parametrize(("n", "expected"), testData)\n\n\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2, skipped=2)

    def test_parametrize_ID_generation_string_int_works(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        '#290'
        pytester.makepyfile("\n            import pytest\n\n            @pytest.fixture\n            def myfixture():\n                return 'example'\n            @pytest.mark.parametrize(\n                'limit', (0, '0'))\n            def test_limit(limit, myfixture):\n                return\n        ")
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    @pytest.mark.parametrize('strict', [True, False])
    def test_parametrize_marked_value(self, pytester: Pytester, strict: bool) -> None:
        if False:
            print('Hello World!')
        s = '\n            import pytest\n\n            @pytest.mark.parametrize(("n", "expected"), [\n                pytest.param(\n                    2,3,\n                    marks=pytest.mark.xfail("sys.version_info > (0, 0, 0)", reason="some bug", strict={strict}),\n                ),\n                pytest.param(\n                    2,3,\n                    marks=[pytest.mark.xfail("sys.version_info > (0, 0, 0)", reason="some bug", strict={strict})],\n                ),\n            ])\n            def test_increment(n, expected):\n                assert n + 1 == expected\n        '.format(strict=strict)
        pytester.makepyfile(s)
        reprec = pytester.inline_run()
        (passed, failed) = (0, 2) if strict else (2, 0)
        reprec.assertoutcome(passed=passed, failed=failed)

    def test_pytest_make_parametrize_id(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            def pytest_make_parametrize_id(config, val):\n                return str(val * 2)\n        ')
        pytester.makepyfile('\n                import pytest\n\n                @pytest.mark.parametrize("x", range(2))\n                def test_func(x):\n                    pass\n                ')
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_func*0*PASS*', '*test_func*2*PASS*'])

    def test_pytest_make_parametrize_id_with_argname(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest("\n            def pytest_make_parametrize_id(config, val, argname):\n                return str(val * 2 if argname == 'x' else val * 10)\n        ")
        pytester.makepyfile('\n                import pytest\n\n                @pytest.mark.parametrize("x", range(2))\n                def test_func_a(x):\n                    pass\n\n                @pytest.mark.parametrize("y", [1])\n                def test_func_b(y):\n                    pass\n                ')
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_func_a*0*PASS*', '*test_func_a*2*PASS*', '*test_func_b*10*PASS*'])

    def test_parametrize_positional_args(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n\n            @pytest.mark.parametrize("a", [1], False)\n            def test_foo(a):\n                pass\n        ')
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

    def test_parametrize_iterator(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import itertools\n            import pytest\n\n            id_parametrize = pytest.mark.parametrize(\n                ids=("param%d" % i for i in itertools.count())\n            )\n\n            @id_parametrize(\'y\', [\'a\', \'b\'])\n            def test1(y):\n                pass\n\n            @id_parametrize(\'y\', [\'a\', \'b\'])\n            def test2(y):\n                pass\n\n            @pytest.mark.parametrize("a, b", [(1, 2), (3, 4)], ids=itertools.count())\n            def test_converted_to_str(a, b):\n                pass\n        ')
        result = pytester.runpytest('-vv', '-s')
        result.stdout.fnmatch_lines(['test_parametrize_iterator.py::test1[param0] PASSED', 'test_parametrize_iterator.py::test1[param1] PASSED', 'test_parametrize_iterator.py::test2[param0] PASSED', 'test_parametrize_iterator.py::test2[param1] PASSED', 'test_parametrize_iterator.py::test_converted_to_str[0] PASSED', 'test_parametrize_iterator.py::test_converted_to_str[1] PASSED', '*= 6 passed in *'])