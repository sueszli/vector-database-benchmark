import subprocess
import textwrap
import pytest
from hypothesistooling.projects.hypothesispython import PYTHON_SRC
from hypothesistooling.scripts import pip_tool, tool_path
PYTHON_VERSIONS = ['3.8', '3.9', '3.10', '3.11']

def test_mypy_passes_on_hypothesis():
    if False:
        i = 10
        return i + 15
    pip_tool('mypy', str(PYTHON_SRC))

@pytest.mark.skip(reason='Hypothesis type-annotates the public API as a convenience for users, but strict checks for our internals would be a net drag on productivity.')
def test_mypy_passes_on_hypothesis_strict():
    if False:
        return 10
    pip_tool('mypy', '--strict', str(PYTHON_SRC))

def get_mypy_output(fname, *extra_args):
    if False:
        return 10
    return subprocess.run([tool_path('mypy'), *extra_args, fname], encoding='utf-8', capture_output=True, text=True).stdout

def get_mypy_analysed_type(fname, val):
    if False:
        for i in range(10):
            print('nop')
    out = get_mypy_output(fname).rstrip()
    msg = 'Success: no issues found in 1 source file'
    if out.endswith(msg):
        out = out[:-len(msg)]
    assert len(out.splitlines()) == 1
    return out.split('Revealed type is ')[1].strip().strip('"' + "'").replace('builtins.', '').replace('*', '').replace('hypothesis.strategies._internal.strategies.SearchStrategy', 'SearchStrategy')

def assert_mypy_errors(fname, expected, python_version=None):
    if False:
        while True:
            i = 10
    _args = ['--no-error-summary', '--show-error-codes']
    if python_version:
        _args.append(f'--python-version={python_version}')
    out = get_mypy_output(fname, *_args)
    del _args

    def convert_lines():
        if False:
            print('Hello World!')
        for error_line in out.splitlines():
            (col, category) = error_line.split(':')[-3:-1]
            if category.strip() != 'error':
                continue
            print(error_line)
            error_code = error_line.split('[')[-1].rstrip(']')
            if error_code == 'empty-body':
                continue
            yield (int(col), error_code)
    assert sorted(convert_lines()) == sorted(expected)

@pytest.mark.parametrize('val,expect', [('integers()', 'int'), ('text()', 'str'), ('integers().map(str)', 'str'), ('booleans().filter(bool)', 'bool'), ('lists(none())', 'list[None]'), ('dictionaries(integers(), datetimes())', 'dict[int, datetime.datetime]'), ('data()', 'hypothesis.strategies._internal.core.DataObject'), ('none() | integers()', 'Union[None, int]'), ('recursive(integers(), lists)', 'Union[list[Ex`-1], int]'), ('one_of(integers(), text())', 'Union[int, str]'), ('one_of(integers(), text(), none(), binary(), builds(list))', 'Union[int, str, None, bytes, list[_T`1]]'), ('one_of(integers(), text(), none(), binary(), builds(list), builds(dict))', 'Any'), ('tuples()', 'tuple[()]'), ('tuples(integers())', 'tuple[int]'), ('tuples(integers(), text())', 'tuple[int, str]'), ('tuples(integers(), text(), integers(), text(), integers())', 'tuple[int, str, int, str, int]'), ('tuples(text(), text(), text(), text(), text(), text())', 'tuple[Any, ...]'), ('from_type(type).flatmap(from_type).filter(lambda x: not isinstance(x, int))', 'Ex_Inv`-1')])
def test_revealed_types(tmpdir, val, expect):
    if False:
        return 10
    'Check that Mypy picks up the expected `X` in SearchStrategy[`X`].'
    f = tmpdir.join(expect + '.py')
    f.write(f'from hypothesis.strategies import *\ns = {val}\nreveal_type(s)\n')
    typ = get_mypy_analysed_type(str(f.realpath()), val)
    assert typ == f'SearchStrategy[{expect}]'

def test_data_object_type_tracing(tmpdir):
    if False:
        return 10
    f = tmpdir.join('check_mypy_on_st_data.py')
    f.write('from hypothesis.strategies import data, integers\nd = data().example()\ns = d.draw(integers())\nreveal_type(s)\n')
    got = get_mypy_analysed_type(str(f.realpath()), 'data().draw(integers())')
    assert got == 'int'

def test_drawfn_type_tracing(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_st_drawfn.py')
    f.write('from hypothesis.strategies import DrawFn, text\ndef comp(draw: DrawFn) -> str:\n    s = draw(text(), 123)\n    reveal_type(s)\n    return s\n')
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == 'str'

def test_composite_type_tracing(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_on_st_composite.py')
    f.write('from hypothesis.strategies import composite, DrawFn\n@composite\ndef comp(draw: DrawFn, x: int) -> int:\n    return x\nreveal_type(comp)\n')
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == 'def (x: int) -> SearchStrategy[int]'

@pytest.mark.parametrize('source, expected', [('', 'def ()'), ('like=f', 'def (x: int) -> int'), ('returns=booleans()', 'def () -> bool'), ('like=f, returns=booleans()', 'def (x: int) -> bool')])
def test_functions_type_tracing(tmpdir, source, expected):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_st_composite.py')
    f.write(f'from hypothesis.strategies import booleans, functions\ndef f(x: int) -> int: return x\ng = functions({source}).example()\nreveal_type(g)\n')
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == expected, (got, expected)

def test_settings_preserves_type(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    f = tmpdir.join('check_mypy_on_settings.py')
    f.write('from hypothesis import settings\n@settings(max_examples=10)\ndef f(x: int) -> int:\n    return x\nreveal_type(f)\n')
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == 'def (x: int) -> int'

def test_stateful_bundle_generic_type(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_stateful_bundle.py')
    f.write("from hypothesis.stateful import Bundle\nb: Bundle[int] = Bundle('test')\nreveal_type(b.example())\n")
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == 'int'

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
@pytest.mark.parametrize('target_args', ['target=b1', 'targets=(b1,)', 'targets=(b1, b2)'])
@pytest.mark.parametrize('returns', ['int', 'MultipleResults[int]'])
def test_stateful_rule_targets(tmpdir, decorator, target_args, returns):
    if False:
        return 10
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write(f"from hypothesis.stateful import *\nb1: Bundle[int] = Bundle('b1')\nb2: Bundle[int] = Bundle('b2')\n@{decorator}({target_args})\ndef my_rule() -> {returns}:\n    ...\n")
    assert_mypy_errors(str(f.realpath()), [])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
def test_stateful_rule_no_targets(tmpdir, decorator):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write(f'from hypothesis.stateful import *\n@{decorator}()\ndef my_rule() -> None:\n    ...\n')
    assert_mypy_errors(str(f.realpath()), [])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
def test_stateful_target_params_mutually_exclusive(tmpdir, decorator):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write(f"from hypothesis.stateful import *\nb1: Bundle[int] = Bundle('b1')\n@{decorator}(target=b1, targets=(b1,))\ndef my_rule() -> int:\n    ...\n")
    assert_mypy_errors(str(f.realpath()), [(3, 'call-overload'), (3, 'misc')])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
@pytest.mark.parametrize('target_args', ['target=b1', 'targets=(b1,)', 'targets=(b1, b2)', ''])
@pytest.mark.parametrize('returns', ['int', 'MultipleResults[int]'])
def test_stateful_target_params_return_type(tmpdir, decorator, target_args, returns):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write(f"from hypothesis.stateful import *\nb1: Bundle[str] = Bundle('b1')\nb2: Bundle[str] = Bundle('b2')\n@{decorator}({target_args})\ndef my_rule() -> {returns}:\n    ...\n")
    assert_mypy_errors(str(f.realpath()), [(4, 'arg-type')])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
def test_stateful_no_target_params_return_type(tmpdir, decorator):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write(f'from hypothesis.stateful import *\n@{decorator}()\ndef my_rule() -> int:\n    ...\n')
    assert_mypy_errors(str(f.realpath()), [(2, 'arg-type')])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
@pytest.mark.parametrize('use_multi', [True, False])
def test_stateful_bundle_variance(tmpdir, decorator, use_multi):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_stateful_bundle.py')
    if use_multi:
        return_type = 'MultipleResults[Dog]'
        return_expr = 'multiple(dog, dog)'
    else:
        return_type = 'Dog'
        return_expr = 'dog'
    f.write(f"from hypothesis.stateful import *\nclass Animal: pass\nclass Dog(Animal): pass\na: Bundle[Animal] = Bundle('animal')\nd: Bundle[Dog] = Bundle('dog')\n@{decorator}(target=a, dog=d)\ndef my_rule(dog: Dog) -> {return_type}:\n    return {return_expr}\n")
    assert_mypy_errors(str(f.realpath()), [])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
def test_stateful_multiple_return(tmpdir, decorator):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_stateful_multiple.py')
    f.write(f"from hypothesis.stateful import *\nb: Bundle[int] = Bundle('b')\n@{decorator}(target=b)\ndef my_rule() -> MultipleResults[int]:\n    return multiple(1, 2, 3)\n")
    assert_mypy_errors(str(f.realpath()), [])

@pytest.mark.parametrize('decorator', ['rule', 'initialize'])
def test_stateful_multiple_return_invalid(tmpdir, decorator):
    if False:
        for i in range(10):
            print('nop')
    f = tmpdir.join('check_mypy_on_stateful_multiple.py')
    f.write(f"from hypothesis.stateful import *\nb: Bundle[str] = Bundle('b')\n@{decorator}(target=b)\ndef my_rule() -> MultipleResults[int]:\n    return multiple(1, 2, 3)\n")
    assert_mypy_errors(str(f.realpath()), [(3, 'arg-type')])

@pytest.mark.parametrize('wrapper,expected', [('{}', 'int'), ('st.lists({})', 'list[int]')])
def test_stateful_consumes_type_tracing(tmpdir, wrapper, expected):
    if False:
        return 10
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    wrapped = wrapper.format('consumes(b)')
    f.write(f"from hypothesis.stateful import *\nfrom hypothesis import strategies as st\nb: Bundle[int] = Bundle('b')\ns = {wrapped}\nreveal_type(s.example())\n")
    got = get_mypy_analysed_type(str(f.realpath()), ...)
    assert got == expected

def test_stateful_consumed_bundle_cannot_be_target(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_on_stateful_rule.py')
    f.write("from hypothesis.stateful import *\nb: Bundle[int] = Bundle('b')\nrule(target=consumes(b))\n")
    assert_mypy_errors(str(f.realpath()), [(3, 'call-overload')])

@pytest.mark.parametrize('return_val,errors', [('True', []), ('0', [(2, 'arg-type'), (2, 'return-value')])])
def test_stateful_precondition_requires_predicate(tmpdir, return_val, errors):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_stateful_precondition.py')
    f.write(f'from hypothesis.stateful import *\n@precondition(lambda self: {return_val})\ndef my_rule() -> None: ...\n')
    assert_mypy_errors(str(f.realpath()), errors)

def test_stateful_precondition_lambda(tmpdir):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_stateful_precondition.py')
    f.write('from hypothesis.stateful import *\nclass MyMachine(RuleBasedStateMachine):\n  valid: bool\n  @precondition(lambda self: self.valid)\n  @rule()\n  def my_rule(self) -> None: ...\n')
    assert_mypy_errors(str(f.realpath()), [])

def test_stateful_precondition_instance_method(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_on_stateful_precondition.py')
    f.write('from hypothesis.stateful import *\nclass MyMachine(RuleBasedStateMachine):\n  valid: bool\n  def check(self) -> bool:\n    return self.valid\n  @precondition(check)\n  @rule()\n  def my_rule(self) -> None: ...\n')
    assert_mypy_errors(str(f.realpath()), [])

def test_stateful_precondition_precond_requires_one_arg(tmpdir):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_stateful_precondition.py')
    f.write('from hypothesis.stateful import *\nprecondition(lambda: True)\nprecondition(lambda a, b: True)\n')
    assert_mypy_errors(str(f.realpath()), [(2, 'arg-type'), (2, 'misc'), (3, 'arg-type'), (3, 'misc')])

def test_pos_only_args(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('check_mypy_on_pos_arg_only_strats.py')
    f.write(textwrap.dedent('\n            import hypothesis.strategies as st\n\n            st.tuples(a1=st.integers())\n            st.tuples(a1=st.integers(), a2=st.integers())\n\n            st.one_of(a1=st.integers())\n            st.one_of(a1=st.integers(), a2=st.integers())\n            '))
    assert_mypy_errors(str(f.realpath()), [(4, 'call-overload'), (5, 'call-overload'), (7, 'call-overload'), (8, 'call-overload')])

@pytest.mark.parametrize('python_version', PYTHON_VERSIONS)
def test_mypy_passes_on_basic_test(tmpdir, python_version):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_basic_tests.py')
    f.write(textwrap.dedent('\n            import hypothesis\n            import hypothesis.strategies as st\n\n            @hypothesis.given(x=st.text())\n            def test_foo(x: str) -> None:\n                assert x == x\n\n            from hypothesis import given\n            from hypothesis.strategies import text\n\n            @given(x=text())\n            def test_bar(x: str) -> None:\n                assert x == x\n            '))
    assert_mypy_errors(str(f.realpath()), [], python_version=python_version)

@pytest.mark.parametrize('python_version', PYTHON_VERSIONS)
def test_given_only_allows_strategies(tmpdir, python_version):
    if False:
        while True:
            i = 10
    f = tmpdir.join('check_mypy_given_expects_strategies.py')
    f.write(textwrap.dedent('\n            from hypothesis import given\n\n            @given(1)\n            def f():\n                pass\n            '))
    assert_mypy_errors(str(f.realpath()), [(4, 'call-overload')], python_version=python_version)

@pytest.mark.parametrize('python_version', PYTHON_VERSIONS)
def test_raises_for_mixed_pos_kwargs_in_given(tmpdir, python_version):
    if False:
        return 10
    f = tmpdir.join('raises_for_mixed_pos_kwargs_in_given.py')
    f.write(textwrap.dedent('\n            from hypothesis import given\n            from hypothesis.strategies import text\n\n            @given(text(), x=text())\n            def test_bar(x):\n                ...\n            '))
    assert_mypy_errors(str(f.realpath()), [(5, 'call-overload')], python_version=python_version)

def test_register_random_interface(tmpdir):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('check_mypy_on_pos_arg_only_strats.py')
    f.write(textwrap.dedent('\n            from random import Random\n            from hypothesis import register_random\n\n            class MyRandom:\n                def __init__(self) -> None:\n                    r = Random()\n                    self.seed = r.seed\n                    self.setstate = r.setstate\n                    self.getstate = r.getstate\n\n            register_random(MyRandom())\n            register_random(None)  # type: ignore[arg-type]\n            '))
    assert_mypy_errors(str(f.realpath()), [])