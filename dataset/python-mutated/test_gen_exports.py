import ast
import sys
from pathlib import Path
import pytest
from trio._tests.pytest_plugin import skip_if_optional_else_raise
try:
    import astor
    import isort
except ImportError as error:
    skip_if_optional_else_raise(error)
from trio._tools.gen_exports import File, create_passthrough_args, get_public_methods, process, run_black, run_linters, run_ruff
SOURCE = 'from _run import _public\nfrom collections import Counter\n\nclass Test:\n    @_public\n    def public_func(self):\n        """With doc string"""\n\n    @ignore_this\n    @_public\n    @another_decorator\n    async def public_async_func(self) -> Counter:\n        pass  # no doc string\n\n    def not_public(self):\n        pass\n\n    async def not_public_async(self):\n        pass\n'
IMPORT_1 = 'from collections import Counter\n'
IMPORT_2 = 'from collections import Counter\nimport os\n'
IMPORT_3 = 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from collections import Counter\n'

def test_get_public_methods() -> None:
    if False:
        for i in range(10):
            print('nop')
    methods = list(get_public_methods(ast.parse(SOURCE)))
    assert {m.name for m in methods} == {'public_func', 'public_async_func'}

def test_create_pass_through_args() -> None:
    if False:
        for i in range(10):
            print('nop')
    testcases = [('def f()', '()'), ('def f(one)', '(one)'), ('def f(one, two)', '(one, two)'), ('def f(one, *args)', '(one, *args)'), ('def f(one, *args, kw1, kw2=None, **kwargs)', '(one, *args, kw1=kw1, kw2=kw2, **kwargs)')]
    for (funcdef, expected) in testcases:
        func_node = ast.parse(funcdef + ':\n  pass').body[0]
        assert isinstance(func_node, ast.FunctionDef)
        assert create_passthrough_args(func_node) == expected
skip_lints = pytest.mark.skipif(sys.implementation.name != 'cpython', reason='gen_exports is internal, black/isort only runs on CPython')

@skip_lints
@pytest.mark.parametrize('imports', [IMPORT_1, IMPORT_2, IMPORT_3])
def test_process(tmp_path: Path, imports: str) -> None:
    if False:
        print('Hello World!')
    try:
        import black
    except ImportError as error:
        skip_if_optional_else_raise(error)
    modpath = tmp_path / '_module.py'
    genpath = tmp_path / '_generated_module.py'
    modpath.write_text(SOURCE, encoding='utf-8')
    file = File(modpath, 'runner', platform='linux', imports=imports)
    assert not genpath.exists()
    with pytest.raises(SystemExit) as excinfo:
        process([file], do_test=True)
    assert excinfo.value.code == 1
    process([file], do_test=False)
    assert genpath.exists()
    process([file], do_test=True)
    with pytest.raises(SystemExit) as excinfo:
        process([File(modpath, 'runner.io_manager', platform='linux', imports=imports)], do_test=True)
    assert excinfo.value.code == 1
    with pytest.raises(SystemExit) as excinfo:
        process([File(modpath, 'runner', imports=imports)], do_test=True)
    assert excinfo.value.code == 1

@skip_lints
def test_run_black(tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    'Test that processing properly fails if black does.'
    try:
        import black
    except ImportError as error:
        skip_if_optional_else_raise(error)
    file = File(tmp_path / 'module.py', 'module')
    (success, _) = run_black(file, 'class not valid code ><')
    assert not success
    (success, _) = run_black(file, 'import waffle\n;import trio')
    assert not success

@skip_lints
def test_run_ruff(tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    'Test that processing properly fails if ruff does.'
    try:
        import ruff
    except ImportError as error:
        skip_if_optional_else_raise(error)
    file = File(tmp_path / 'module.py', 'module')
    (success, _) = run_ruff(file, 'class not valid code ><')
    assert not success
    test_function = 'def combine_and(data: list[str]) -> str:\n    """Join values of text, and have \'and\' with the last one properly."""\n    if len(data) >= 2:\n        data[-1] = \'and \' + data[-1]\n    if len(data) > 2:\n        return \', \'.join(data)\n    return \' \'.join(data)'
    (success, response) = run_ruff(file, test_function)
    assert success
    assert response == test_function

@skip_lints
def test_lint_failure(tmp_path: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test that processing properly fails if black or ruff does.'
    try:
        import black
        import ruff
    except ImportError as error:
        skip_if_optional_else_raise(error)
    file = File(tmp_path / 'module.py', 'module')
    with pytest.raises(SystemExit):
        run_linters(file, 'class not valid code ><')
    with pytest.raises(SystemExit):
        run_linters(file, 'import waffle\n;import trio')