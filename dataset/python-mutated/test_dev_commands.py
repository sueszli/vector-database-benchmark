import sys
import textwrap
from typing import Any, Dict, Optional
from unittest.mock import MagicMock
import pytest
from redbot.core import commands
from redbot.core.dev_commands import DevOutput, SourceCache, cleanup_code

@pytest.mark.parametrize('content,source', (('x = 1', 'x = 1'), ('`x = 1`', 'x = 1'), ('``x = 1``', 'x = 1'), ('```x = 1```', 'x = 1'), ('            ```x = 1\n            ```', 'x = 1'), ('            ```\n            x = 1```', 'x = 1'), ('            ```\n            x = 1\n            ```', 'x = 1'), ('            ```py\n            x = 1\n            ```', 'x = 1'), ('            ```python\n            x = 1\n            ```', 'x = 1'), ('            ```py\n            x = 1```', 'x = 1'), ('            ```python\n            x = 1```', 'x = 1'), ('            ```pass\n            ```', 'pass'), ('            ```\n\n\n            x = 1\n            ```', 'x = 1'), ('            ```python\n\n\n            x = 1\n            ```', 'x = 1'), ('            ```\n\n\n            x = 1```', 'x = 1'), ('            ```python\n\n\n            x = 1```', 'x = 1')))
def test_cleanup_code(content: str, source: str) -> None:
    if False:
        print('Hello World!')
    content = textwrap.dedent(content)
    source = textwrap.dedent(source)
    assert cleanup_code(content) == source

def _get_dev_output(source: str, *, source_cache: Optional[SourceCache]=None, env: Optional[Dict[str, Any]]=None) -> DevOutput:
    if False:
        i = 10
        return i + 15
    if source_cache is None:
        source_cache = SourceCache()
    return DevOutput(MagicMock(spec=commands.Context), source_cache=source_cache, filename=f'<test run - snippet #{source_cache.take_next_index()}>', source=source, env={'__builtins__': __builtins__, '__name__': '__main__', '_': None, **(env or {})})

async def _run_dev_output(monkeypatch: pytest.MonkeyPatch, source: str, result: str, *, debug: bool=False, eval: bool=False, repl: bool=False) -> None:
    source = textwrap.dedent(source)
    result = textwrap.dedent(result)
    monkeypatch.setattr('redbot.core.dev_commands.sanitize_output', lambda ctx, s: s)
    if debug:
        output = _get_dev_output(source)
        await output.run_debug()
        assert str(output) == result
        assert not output.ctx.mock_calls
    if eval:
        output = _get_dev_output(source.replace('<module>', 'func'))
        await output.run_eval()
        assert str(output) == result.replace('<module>', 'func')
        assert not output.ctx.mock_calls
    if repl:
        output = _get_dev_output(source)
        await output.run_repl()
        assert str(output) == result
        assert not output.ctx.mock_calls
EXPRESSION_TESTS = {'12x\n': ((lambda v: v < (3, 10), '              File "<test run - snippet #0>", line 1\n                12x\n                  ^\n            SyntaxError: invalid syntax\n            '), (lambda v: v >= (3, 10), '              File "<test run - snippet #0>", line 1\n                12x\n                 ^\n            SyntaxError: invalid decimal literal\n            ')), 'foo(x, z for z in range(10), t, w)': ((lambda v: v < (3, 10), '              File "<test run - snippet #0>", line 1\n                foo(x, z for z in range(10), t, w)\n                       ^\n            SyntaxError: Generator expression must be parenthesized\n            '), (lambda v: v >= (3, 10), '              File "<test run - snippet #0>", line 1\n                foo(x, z for z in range(10), t, w)\n                       ^^^^^^^^^^^^^^^^^^^^\n            SyntaxError: Generator expression must be parenthesized\n            ')), 'abs(1 / 0)': ((lambda v: v < (3, 11), '            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 1, in <module>\n                abs(1 / 0)\n            ZeroDivisionError: division by zero\n            '), (lambda v: v >= (3, 11), '            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 1, in <module>\n                abs(1 / 0)\n                    ~~^~~\n            ZeroDivisionError: division by zero\n            '))}
STATEMENT_TESTS = {'    def x():\n        12x\n    ': ((lambda v: v < (3, 10), '              File "<test run - snippet #0>", line 2\n                12x\n                  ^\n            SyntaxError: invalid syntax\n            '), (lambda v: v >= (3, 10), '              File "<test run - snippet #0>", line 2\n                12x\n                 ^\n            SyntaxError: invalid decimal literal\n            ')), '    def x():\n        foo(x, z for z in range(10), t, w)\n    ': ((lambda v: v < (3, 10), '              File "<test run - snippet #0>", line 2\n                foo(x, z for z in range(10), t, w)\n                       ^\n            SyntaxError: Generator expression must be parenthesized\n            '), (lambda v: v >= (3, 10), '              File "<test run - snippet #0>", line 2\n                foo(x, z for z in range(10), t, w)\n                       ^^^^^^^^^^^^^^^^^^^^\n            SyntaxError: Generator expression must be parenthesized\n            ')), '    print(123)\n    try:\n        abs(1 / 0)\n    except ValueError:\n        pass\n    ': ((lambda v: v < (3, 11), '            123\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 3, in <module>\n                abs(1 / 0)\n            ZeroDivisionError: division by zero\n            '), (lambda v: v >= (3, 11), '            123\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 3, in <module>\n                abs(1 / 0)\n                    ~~^~~\n            ZeroDivisionError: division by zero\n            ')), '    try:\n        1 / 0\n    except ZeroDivisionError as exc:\n        try:\n            raise RuntimeError("direct cause") from exc\n        except RuntimeError:\n            raise ValueError("indirect cause")\n    ': ((lambda v: v < (3, 11), '            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 2, in <module>\n                1 / 0\n            ZeroDivisionError: division by zero\n\n            The above exception was the direct cause of the following exception:\n\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 5, in <module>\n                raise RuntimeError("direct cause") from exc\n            RuntimeError: direct cause\n\n            During handling of the above exception, another exception occurred:\n\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 7, in <module>\n                raise ValueError("indirect cause")\n            ValueError: indirect cause\n            '), (lambda v: v >= (3, 11), '            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 2, in <module>\n                1 / 0\n                ~~^~~\n            ZeroDivisionError: division by zero\n\n            The above exception was the direct cause of the following exception:\n\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 5, in <module>\n                raise RuntimeError("direct cause") from exc\n            RuntimeError: direct cause\n\n            During handling of the above exception, another exception occurred:\n\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 7, in <module>\n                raise ValueError("indirect cause")\n            ValueError: indirect cause\n            ')), '    def f(v):\n        try:\n            1 / 0\n        except ZeroDivisionError:\n            try:\n                raise ValueError(v)\n            except ValueError as e:\n                return e\n    try:\n        raise ExceptionGroup("one", [f(1)])\n    except ExceptionGroup as e:\n        eg = e\n    try:\n        raise ExceptionGroup("two", [f(2), eg])\n    except ExceptionGroup as e:\n        raise RuntimeError("wrapping") from e\n    ': ((lambda v: v >= (3, 11), '              + Exception Group Traceback (most recent call last):\n              |   File "<test run - snippet #0>", line 14, in <module>\n              |     raise ExceptionGroup("two", [f(2), eg])\n              | ExceptionGroup: two (2 sub-exceptions)\n              +-+---------------- 1 ----------------\n                | Traceback (most recent call last):\n                |   File "<test run - snippet #0>", line 3, in f\n                |     1 / 0\n                |     ~~^~~\n                | ZeroDivisionError: division by zero\n                | \n                | During handling of the above exception, another exception occurred:\n                | \n                | Traceback (most recent call last):\n                |   File "<test run - snippet #0>", line 6, in f\n                |     raise ValueError(v)\n                | ValueError: 2\n                +---------------- 2 ----------------\n                | Exception Group Traceback (most recent call last):\n                |   File "<test run - snippet #0>", line 10, in <module>\n                |     raise ExceptionGroup("one", [f(1)])\n                | ExceptionGroup: one (1 sub-exception)\n                +-+---------------- 1 ----------------\n                  | Traceback (most recent call last):\n                  |   File "<test run - snippet #0>", line 3, in f\n                  |     1 / 0\n                  |     ~~^~~\n                  | ZeroDivisionError: division by zero\n                  | \n                  | During handling of the above exception, another exception occurred:\n                  | \n                  | Traceback (most recent call last):\n                  |   File "<test run - snippet #0>", line 6, in f\n                  |     raise ValueError(v)\n                  | ValueError: 1\n                  +------------------------------------\n\n            The above exception was the direct cause of the following exception:\n\n            Traceback (most recent call last):\n              File "<test run - snippet #0>", line 16, in <module>\n                raise RuntimeError("wrapping") from e\n            RuntimeError: wrapping\n            '),)}

@pytest.mark.parametrize('source,result', [(source, result) for (source, results) in EXPRESSION_TESTS.items() for (condition, result) in results if condition(sys.version_info)])
async def test_format_exception_expressions(monkeypatch: pytest.MonkeyPatch, source: str, result: str) -> None:
    await _run_dev_output(monkeypatch, source, result, debug=True, repl=True)

@pytest.mark.parametrize('source,result', [(source, result) for (source, results) in STATEMENT_TESTS.items() for (condition, result) in results if condition(sys.version_info)])
async def test_format_exception_statements(monkeypatch: pytest.MonkeyPatch, source: str, result: str) -> None:
    await _run_dev_output(monkeypatch, source, result, eval=True, repl=True)

async def test_successful_run_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "print('hello world'), 123"
    result = '(None, 123)'
    await _run_dev_output(monkeypatch, source, result, debug=True)

async def test_successful_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    source = '    print("hello world")\n    return 123\n    '
    result = '    hello world\n    123'
    await _run_dev_output(monkeypatch, source, result, eval=True)

async def test_successful_run_repl_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "print('hello world'), 123"
    result = '    hello world\n    (None, 123)'
    await _run_dev_output(monkeypatch, source, result, repl=True)

async def test_successful_run_repl_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    source = '    print("hello")\n    print("world")\n    '
    result = '    hello\n    world\n    '
    await _run_dev_output(monkeypatch, source, result, repl=True)

async def test_regression_format_exception_from_previous_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    snippet_0 = textwrap.dedent('    def repro():\n        raise Exception("this is an error!")\n\n    return repro\n    ')
    snippet_1 = '_()'
    result = textwrap.dedent('    Traceback (most recent call last):\n      File "<test run - snippet #1>", line 1, in func\n        _()\n      File "<test run - snippet #0>", line 2, in repro\n        raise Exception("this is an error!")\n    Exception: this is an error!\n    ')
    monkeypatch.setattr('redbot.core.dev_commands.sanitize_output', lambda ctx, s: s)
    source_cache = SourceCache()
    output = _get_dev_output(snippet_0, source_cache=source_cache)
    await output.run_eval()
    output = _get_dev_output(snippet_1, source_cache=source_cache, env={'_': output.result})
    await output.run_eval()
    assert str(output) == result
    assert not output.ctx.mock_calls