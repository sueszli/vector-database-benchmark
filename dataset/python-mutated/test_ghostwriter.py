import ast
import enum
import json
import re
import socket
import unittest
import unittest.mock
from decimal import Decimal
from pathlib import Path
from textwrap import dedent
from types import FunctionType, ModuleType
from typing import Any, FrozenSet, KeysView, List, Match, Pattern, Sequence, Set, Sized, Union, ValuesView
import attr
import click
import pytest
from hypothesis import HealthCheck, assume, settings
from hypothesis.errors import InvalidArgument, Unsatisfiable
from hypothesis.extra import cli, ghostwriter
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.strategies import builds, from_type, just, lists
from hypothesis.strategies._internal.core import from_regex
from hypothesis.strategies._internal.lazy import LazyStrategy
varied_excepts = pytest.mark.parametrize('ex', [(), ValueError, (TypeError, re.error)])

def get_test_function(source_code, settings_decorator=lambda fn: fn):
    if False:
        i = 10
        return i + 15
    namespace = {}
    try:
        exec(source_code, namespace)
    except Exception:
        print(f'************\n{source_code}\n************')
        raise
    tests = [v for (k, v) in namespace.items() if k.startswith(('test_', 'Test')) and (not isinstance(v, ModuleType))]
    assert len(tests) == 1, tests
    return settings_decorator(tests[0])

@pytest.mark.parametrize('badness', ['not an exception', BaseException, [ValueError], (Exception, 'bad')])
def test_invalid_exceptions(badness):
    if False:
        i = 10
        return i + 15
    with pytest.raises(InvalidArgument):
        ghostwriter._check_except(badness)

def test_style_validation():
    if False:
        for i in range(10):
            print('nop')
    ghostwriter._check_style('pytest')
    ghostwriter._check_style('unittest')
    with pytest.raises(InvalidArgument):
        ghostwriter._check_style('not a valid style')

def test_strategies_with_invalid_syntax_repr_as_nothing():
    if False:
        for i in range(10):
            print('nop')
    msg = '$$ this repr is not Python syntax $$'

    class NoRepr:

        def __repr__(self):
            if False:
                print('Hello World!')
            return msg
    s = just(NoRepr())
    assert repr(s) == f'just({msg})'
    assert ghostwriter._valid_syntax_repr(s)[1] == 'nothing()'

class AnEnum(enum.Enum):
    a = 'value of AnEnum.a'
    b = 'value of AnEnum.b'

def takes_enum(foo=AnEnum.a):
    if False:
        print('Hello World!')
    assert foo != AnEnum.b

def test_ghostwriter_exploits_arguments_with_enum_defaults():
    if False:
        i = 10
        return i + 15
    source_code = ghostwriter.fuzz(takes_enum)
    test = get_test_function(source_code)
    with pytest.raises(AssertionError):
        test()

def timsort(seq: Sequence[int]) -> List[int]:
    if False:
        i = 10
        return i + 15
    return sorted(seq)

def non_type_annotation(x: 3):
    if False:
        print('Hello World!')
    pass

def annotated_any(x: Any):
    if False:
        while True:
            i = 10
    pass
space_in_name = type('a name', (type,), {'__init__': lambda self: None})

class NotResolvable:

    def __init__(self, unannotated_required):
        if False:
            for i in range(10):
                print('nop')
        pass

def non_resolvable_arg(x: NotResolvable):
    if False:
        return 10
    pass

def test_flattens_one_of_repr():
    if False:
        for i in range(10):
            print('nop')
    strat = from_type(Union[int, Sequence[int]])
    assert repr(strat).count('one_of(') > 1
    assert ghostwriter._valid_syntax_repr(strat)[1].count('one_of(') == 1

def takes_keys(x: KeysView[int]) -> None:
    if False:
        while True:
            i = 10
    pass

def takes_values(x: ValuesView[int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def takes_match(x: Match[bytes]) -> None:
    if False:
        i = 10
        return i + 15
    pass

def takes_pattern(x: Pattern[str]) -> None:
    if False:
        i = 10
        return i + 15
    pass

def takes_sized(x: Sized) -> None:
    if False:
        print('Hello World!')
    pass

def takes_frozensets(a: FrozenSet[int], b: FrozenSet[int]) -> None:
    if False:
        return 10
    pass

@attr.s()
class Foo:
    foo: str = attr.ib()

def takes_attrs_class(x: Foo) -> None:
    if False:
        return 10
    pass

@varied_excepts
@pytest.mark.parametrize('func', [re.compile, json.loads, json.dump, timsort, ast.literal_eval, non_type_annotation, annotated_any, space_in_name, non_resolvable_arg, takes_keys, takes_values, takes_match, takes_pattern, takes_sized, takes_frozensets, takes_attrs_class])
def test_ghostwriter_fuzz(func, ex):
    if False:
        i = 10
        return i + 15
    source_code = ghostwriter.fuzz(func, except_=ex)
    get_test_function(source_code)

def test_socket_module():
    if False:
        while True:
            i = 10
    source_code = ghostwriter.magic(socket)
    exec(source_code, {})

def test_binary_op_also_handles_frozensets():
    if False:
        i = 10
        return i + 15
    source_code = ghostwriter.binary_operation(takes_frozensets)
    exec(source_code, {})

@varied_excepts
@pytest.mark.parametrize('func', [re.compile, json.loads, json.dump, timsort, ast.literal_eval])
def test_ghostwriter_unittest_style(func, ex):
    if False:
        for i in range(10):
            print('nop')
    source_code = ghostwriter.fuzz(func, except_=ex, style='unittest')
    assert issubclass(get_test_function(source_code), unittest.TestCase)

def no_annotations(foo=None, *, bar=False):
    if False:
        print('Hello World!')
    pass

def test_inference_from_defaults_and_none_booleans_reprs_not_just_and_sampled_from():
    if False:
        for i in range(10):
            print('nop')
    source_code = ghostwriter.fuzz(no_annotations)
    assert '@given(foo=st.none(), bar=st.booleans())' in source_code

def hopefully_hashable(foo: Set[Decimal]):
    if False:
        while True:
            i = 10
    pass

def test_no_hashability_filter():
    if False:
        for i in range(10):
            print('nop')
    source_code = ghostwriter.fuzz(hopefully_hashable)
    assert '@given(foo=st.sets(st.decimals()))' in source_code
    assert '_can_hash' not in source_code

@pytest.mark.parametrize('gw,args', [(ghostwriter.fuzz, ['not callable']), (ghostwriter.idempotent, ['not callable']), (ghostwriter.roundtrip, []), (ghostwriter.roundtrip, ['not callable']), (ghostwriter.equivalent, [sorted]), (ghostwriter.equivalent, [sorted, 'not callable'])])
def test_invalid_func_inputs(gw, args):
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidArgument):
        gw(*args)

class A:

    @classmethod
    def to_json(cls, obj: Union[dict, list]) -> str:
        if False:
            print('Hello World!')
        return json.dumps(obj)

    @classmethod
    def from_json(cls, obj: str) -> Union[dict, list]:
        if False:
            while True:
                i = 10
        return json.loads(obj)

    @staticmethod
    def static_sorter(seq: Sequence[int]) -> List[int]:
        if False:
            print('Hello World!')
        return sorted(seq)

@pytest.mark.parametrize('gw,args', [(ghostwriter.fuzz, [A.static_sorter]), (ghostwriter.idempotent, [A.static_sorter]), (ghostwriter.roundtrip, [A.to_json, A.from_json]), (ghostwriter.equivalent, [A.to_json, json.dumps])])
def test_class_methods_inputs(gw, args):
    if False:
        i = 10
        return i + 15
    source_code = gw(*args)
    get_test_function(source_code)()

def test_run_ghostwriter_fuzz():
    if False:
        return 10
    source_code = ghostwriter.fuzz(sorted)
    assert 'st.nothing()' not in source_code
    get_test_function(source_code)()

class MyError(UnicodeDecodeError):
    pass

@pytest.mark.parametrize('exceptions,output', [((Exception, UnicodeError), 'Exception'), ((UnicodeError, MyError), 'UnicodeError'), ((IOError,), 'OSError'), ((IOError, UnicodeError), '(OSError, UnicodeError)')])
def test_exception_deduplication(exceptions, output):
    if False:
        i = 10
        return i + 15
    (_, body) = ghostwriter._make_test_body(lambda : None, ghost='', test_body='pass', except_=exceptions, style='pytest', annotate=False)
    assert f'except {output}:' in body

def test_run_ghostwriter_roundtrip():
    if False:
        while True:
            i = 10
    source_code = ghostwriter.roundtrip(json.dumps, json.loads)
    with pytest.raises(Unsatisfiable):
        get_test_function(source_code)()
    source_code = source_code.replace('st.nothing()', 'st.recursive(st.one_of(st.none(), st.booleans(), st.floats(), st.text()), lambda v: st.lists(v, max_size=2) | st.dictionaries(st.text(), v, max_size=2), max_leaves=2)')
    s = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    try:
        get_test_function(source_code, settings_decorator=s)()
    except (AssertionError, ValueError, BaseExceptionGroup):
        pass
    source_code = source_code.replace('st.floats()', 'st.floats(allow_nan=False, allow_infinity=False)')
    get_test_function(source_code, settings_decorator=s)()

@varied_excepts
@pytest.mark.parametrize('func', [sorted, timsort])
def test_ghostwriter_idempotent(func, ex):
    if False:
        print('Hello World!')
    source_code = ghostwriter.idempotent(func, except_=ex)
    test = get_test_function(source_code)
    if '=st.nothing()' in source_code:
        with pytest.raises(Unsatisfiable):
            test()
    else:
        test()

def test_overlapping_args_use_union_of_strategies():
    if False:
        i = 10
        return i + 15

    def f(arg: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def g(arg: float) -> None:
        if False:
            return 10
        pass
    source_code = ghostwriter.equivalent(f, g)
    assert 'arg=st.one_of(st.integers(), st.floats())' in source_code

def test_module_with_mock_does_not_break():
    if False:
        for i in range(10):
            print('nop')
    ghostwriter.magic(unittest.mock)

def compose_types(x: type, y: type):
    if False:
        print('Hello World!')
    pass

def test_unrepr_identity_elem():
    if False:
        i = 10
        return i + 15
    source_code = ghostwriter.binary_operation(compose_types)
    exec(source_code, {})
    source_code = ghostwriter.binary_operation(compose_types, identity=type)
    exec(source_code, {})

@pytest.mark.parametrize('strategy, imports', [(LazyStrategy(from_type, (enum.Enum,), {}), {('enum', 'Enum')}), (builds(enum.Enum).map(Decimal), {('enum', 'Enum'), ('decimal', 'Decimal')}), (builds(enum.Enum).flatmap(Decimal), {('enum', 'Enum'), ('decimal', 'Decimal')}), (builds(enum.Enum).filter(Decimal).filter(re.compile), {('enum', 'Enum'), ('decimal', 'Decimal'), ('re', 'compile')}), (builds(enum.Enum) | builds(Decimal) | builds(re.compile), {('enum', 'Enum'), ('decimal', 'Decimal'), ('re', 'compile')}), (builds(enum.Enum, builds(Decimal), kw=builds(re.compile)), {('enum', 'Enum'), ('decimal', 'Decimal'), ('re', 'compile')}), (lists(builds(Decimal)), {('decimal', 'Decimal')}), (from_regex(re.compile('.+')), {'re'}), (from_regex('.+'), set())])
def test_get_imports_for_strategy(strategy, imports):
    if False:
        print('Hello World!')
    assert ghostwriter._imports_for_strategy(strategy) == imports

@pytest.fixture
def temp_script_file():
    if False:
        while True:
            i = 10
    'Fixture to yield a Path to a temporary file in the local directory. File name will end\n    in .py and will include an importable function.\n    '
    p = Path('my_temp_script.py')
    if p.exists():
        raise FileExistsError(f'Did not expect {p} to exist during testing')
    p.write_text(dedent('\n            def say_hello():\n                print("Hello world!")\n            '), encoding='utf-8')
    yield p
    p.unlink()

@pytest.fixture
def temp_script_file_with_py_function():
    if False:
        while True:
            i = 10
    'Fixture to yield a Path to a temporary file in the local directory. File name will end\n    in .py and will include an importable function named "py"\n    '
    p = Path('my_temp_script_with_py_function.py')
    if p.exists():
        raise FileExistsError(f'Did not expect {p} to exist during testing')
    p.write_text(dedent('\n            def py():\n                print(\'A function named "py" has been called\')\n            '), encoding='utf-8')
    yield p
    p.unlink()

def test_obj_name(temp_script_file, temp_script_file_with_py_function):
    if False:
        while True:
            i = 10
    with pytest.raises(click.exceptions.UsageError) as e:
        cli.obj_name('mydirectory/myscript.py')
    assert e.match('Remember that the ghostwriter should be passed the name of a module, not a path.')
    with pytest.raises(click.exceptions.UsageError) as e:
        cli.obj_name('mydirectory\\myscript.py')
    assert e.match('Remember that the ghostwriter should be passed the name of a module, not a path.')
    with pytest.raises(click.exceptions.UsageError) as e:
        cli.obj_name('myscript.py')
    assert e.match('Remember that the ghostwriter should be passed the name of a module, not a file.')
    with pytest.raises(click.exceptions.UsageError) as e:
        cli.obj_name(str(temp_script_file))
    assert e.match(f'Remember that the ghostwriter should be passed the name of a module, not a file.\n\tTry: hypothesis write {temp_script_file.stem}')
    assert isinstance(cli.obj_name(str(temp_script_file_with_py_function)), FunctionType)

def test_gets_public_location_not_impl_location():
    if False:
        for i in range(10):
            print('nop')
    assert ghostwriter._get_module(assume) == 'hypothesis'