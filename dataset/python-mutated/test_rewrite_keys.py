import sys
from pathlib import Path
from textwrap import dedent
import pytest
from click.testing import CliRunner
from reactpy._console.rewrite_keys import generate_rewrite, rewrite_keys
if sys.version_info < (3, 9):
    pytestmark = pytest.mark.skip(reason='ast.unparse is Python>=3.9')

def test_rewrite_key_declarations(tmp_path):
    if False:
        print('Hello World!')
    runner = CliRunner()
    tempfile: Path = tmp_path / 'temp.py'
    tempfile.write_text("html.div(key='test')")
    result = runner.invoke(rewrite_keys, args=[str(tmp_path)], catch_exceptions=False)
    assert result.exit_code == 0
    assert tempfile.read_text() == "html.div({'key': 'test'})"

def test_rewrite_key_declarations_no_files():
    if False:
        while True:
            i = 10
    runner = CliRunner()
    result = runner.invoke(rewrite_keys, args=['directory-does-no-exist'], catch_exceptions=False)
    assert result.exit_code != 0

@pytest.mark.parametrize('source, expected', [("html.div(key='test')", "html.div({'key': 'test'})"), ("html.div('something', key='test')", "html.div({'key': 'test'}, 'something')"), ("html.div({'some_attr': 1}, child_1, child_2, key='test')", "html.div({'some_attr': 1, 'key': 'test'}, child_1, child_2)"), ("vdom('div', key='test')", "vdom('div', {'key': 'test'})"), ("vdom('div', 'something', key='test')", "vdom('div', {'key': 'test'}, 'something')"), ("vdom('div', {'some_attr': 1}, child_1, child_2, key='test')", "vdom('div', {'some_attr': 1, 'key': 'test'}, child_1, child_2)"), ("html.div(dict(some_attr=1), child_1, child_2, key='test')", "html.div(dict(some_attr=1, key='test'), child_1, child_2)"), ("vdom('div', dict(some_attr=1), child_1, child_2, key='test')", "vdom('div', dict(some_attr=1, key='test'), child_1, child_2)"), ("\n            def my_function():\n                x = 1  # some comment\n                return html.div(key='test')\n            ", "\n            def my_function():\n                x = 1  # some comment\n                return html.div({'key': 'test'})\n            "), ("\n            if condition:\n                # some comment\n                dom = html.div(key='test')\n            ", "\n            if condition:\n                # some comment\n                dom = html.div({'key': 'test'})\n            "), ("\n            [\n                html.div(key='test'),\n                html.div(key='test'),\n            ]\n            ", "\n            [\n                html.div({'key': 'test'}),\n                html.div({'key': 'test'}),\n            ]\n            "), ("\n            @deco(\n                html.div(key='test'),\n                html.div(key='test'),\n            )\n            def func():\n                # comment\n                x = [\n                    1\n                ]\n            ", "\n            @deco(\n                html.div({'key': 'test'}),\n                html.div({'key': 'test'}),\n            )\n            def func():\n                # comment\n                x = [\n                    1\n                ]\n            "), ("\n            @deco(html.div(key='test'), html.div(key='test'))\n            def func():\n                # comment\n                x = [\n                    1\n                ]\n            ", "\n            @deco(html.div({'key': 'test'}), html.div({'key': 'test'}))\n            def func():\n                # comment\n                x = [\n                    1\n                ]\n            "), ("\n            (\n                result\n                if condition\n                else html.div(key='test')\n            )\n            ", "\n            (\n                result\n                if condition\n                else html.div({'key': 'test'})\n            )\n            "), ('\n            x = 1\n            html.div(\n                "hello",\n                # comment 1\n                html.div(key=\'test\'),\n                # comment 2\n                key=\'test\',\n            )\n            ', "\n            x = 1\n            # comment 1\n            # comment 2\n            html.div({'key': 'test'}, 'hello', html.div({'key': 'test'}))\n            "), ("html.no_an_element(key='test')", None), ("not_html.div(key='test')", None), ('html.div()', None), ("html.div(not_key='something')", None), ('vdom()', None), ("(some + expr)(key='test')", None), ('html.div()', None), ("html.div(child_1, child_2, key='test')", None), ("vdom('div', child_1, child_2, key='test')", None)], ids=lambda item: ' '.join(map(str.strip, item.split())) if isinstance(item, str) else item)
def test_generate_rewrite(source, expected):
    if False:
        i = 10
        return i + 15
    actual = generate_rewrite(Path('test.py'), dedent(source).strip())
    if isinstance(expected, str):
        expected = dedent(expected).strip()
    assert actual == expected