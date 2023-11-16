from __future__ import annotations
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any
import pytest
from hypothesistooling.projects.hypothesispython import HYPOTHESIS_PYTHON, PYTHON_SRC
from hypothesistooling.scripts import pip_tool, tool_path
PYTHON_VERSIONS = ['3.7', '3.8', '3.9', '3.10', '3.11']

@pytest.mark.skip(reason='Hypothesis type-annotates the public API as a convenience for users, but strict checks for our internals would be a net drag on productivity.')
def test_pyright_passes_on_hypothesis():
    if False:
        print('Hello World!')
    pip_tool('pyright', '--project', HYPOTHESIS_PYTHON)

@pytest.mark.parametrize('python_version', PYTHON_VERSIONS)
def test_pyright_passes_on_basic_test(tmp_path: Path, python_version: str):
    if False:
        while True:
            i = 10
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            import hypothesis\n            import hypothesis.strategies as st\n\n            @hypothesis.given(x=st.text())\n            def test_foo(x: str):\n                assert x == x\n\n            from hypothesis import given\n            from hypothesis.strategies import text\n\n            @given(x=text())\n            def test_bar(x: str):\n                assert x == x\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict', 'pythonVersion': python_version})
    assert _get_pyright_errors(file) == []

@pytest.mark.parametrize('python_version', PYTHON_VERSIONS)
def test_given_only_allows_strategies(tmp_path: Path, python_version: str):
    if False:
        return 10
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            from hypothesis import given\n\n            @given(1)\n            def f():\n                pass\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict', 'pythonVersion': python_version})
    assert sum((e['message'].startswith('Argument of type "Literal[1]" cannot be assigned to parameter "_given_arguments"') for e in _get_pyright_errors(file))) == 1

def test_pyright_issue_3296(tmp_path: Path):
    if False:
        while True:
            i = 10
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            from hypothesis.strategies import lists, integers\n\n            lists(integers()).map(sorted)\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict'})
    assert _get_pyright_errors(file) == []

def test_pyright_raises_for_mixed_pos_kwargs_in_given(tmp_path: Path):
    if False:
        while True:
            i = 10
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            from hypothesis import given\n            from hypothesis.strategies import text\n\n            @given(text(), x=text())\n            def test_bar(x: str):\n                pass\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict'})
    assert sum((e['message'].startswith('No overloads for "given" match the provided arguments') for e in _get_pyright_errors(file))) == 1

def test_pyright_issue_3348(tmp_path: Path):
    if False:
        for i in range(10):
            print('nop')
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            import hypothesis.strategies as st\n\n            st.tuples(st.integers(), st.integers())\n            st.one_of(st.integers(), st.integers())\n            st.one_of([st.integers(), st.floats()])  # sequence of strats should be OK\n            st.sampled_from([1, 2])\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict'})
    assert _get_pyright_errors(file) == []

def test_pyright_tuples_pos_args_only(tmp_path: Path):
    if False:
        while True:
            i = 10
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            import hypothesis.strategies as st\n\n            st.tuples(a1=st.integers())\n            st.tuples(a1=st.integers(), a2=st.integers())\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict'})
    assert sum((e['message'].startswith('No overloads for "tuples" match the provided arguments') for e in _get_pyright_errors(file))) == 2

def test_pyright_one_of_pos_args_only(tmp_path: Path):
    if False:
        i = 10
        return i + 15
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            import hypothesis.strategies as st\n\n            st.one_of(a1=st.integers())\n            st.one_of(a1=st.integers(), a2=st.integers())\n            '), encoding='utf-8')
    _write_config(tmp_path, {'typeCheckingMode': 'strict'})
    assert sum((e['message'].startswith('No overloads for "one_of" match the provided arguments') for e in _get_pyright_errors(file))) == 2

def test_register_random_protocol(tmp_path: Path):
    if False:
        for i in range(10):
            print('nop')
    file = tmp_path / 'test.py'
    file.write_text(textwrap.dedent('\n            from random import Random\n            from hypothesis import register_random\n\n            class MyRandom:\n                def __init__(self) -> None:\n                    r = Random()\n                    self.seed = r.seed\n                    self.setstate = r.setstate\n                    self.getstate = r.getstate\n\n            register_random(MyRandom())\n            register_random(None)  # type: ignore\n            '), encoding='utf-8')
    _write_config(tmp_path, {'reportUnnecessaryTypeIgnoreComment': True})
    assert _get_pyright_errors(file) == []

def _get_pyright_output(file: Path) -> dict[str, Any]:
    if False:
        print('Hello World!')
    proc = subprocess.run([tool_path('pyright'), '--outputjson'], cwd=file.parent, encoding='utf-8', text=True, capture_output=True)
    try:
        return json.loads(proc.stdout)
    except Exception:
        print(proc.stdout)
        raise

def _get_pyright_errors(file: Path) -> list[dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    return _get_pyright_output(file)['generalDiagnostics']

def _write_config(config_dir: Path, data: dict[str, Any] | None=None):
    if False:
        return 10
    config = {'extraPaths': [str(PYTHON_SRC)], **(data or {})}
    (config_dir / 'pyrightconfig.json').write_text(json.dumps(config), encoding='utf-8')