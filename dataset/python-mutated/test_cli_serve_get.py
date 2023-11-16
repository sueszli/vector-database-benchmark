from datasette.cli import cli, serve
from datasette.plugins import pm
from click.testing import CliRunner
import textwrap
import json

def test_serve_with_get(tmp_path_factory):
    if False:
        while True:
            i = 10
    plugins_dir = tmp_path_factory.mktemp('plugins_for_serve_with_get')
    (plugins_dir / 'init_for_serve_with_get.py').write_text(textwrap.dedent('\n        from datasette import hookimpl\n\n        @hookimpl\n        def startup(datasette):\n            with open("{}", "w") as fp:\n                fp.write("hello")\n    '.format(str(plugins_dir / 'hello.txt'))), 'utf-8')
    runner = CliRunner()
    result = runner.invoke(cli, ['serve', '--memory', '--plugins-dir', str(plugins_dir), '--get', '/_memory.json?sql=select+sqlite_version()'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data['rows']) == 1
    assert list(data['rows'][0].keys()) == ['sqlite_version()']
    assert set(data.keys()) == {'rows', 'ok', 'truncated'}
    assert (plugins_dir / 'hello.txt').read_text() == 'hello'
    to_unregister = [p for p in pm.get_plugins() if p.__name__ == 'init_for_serve_with_get.py'][0]
    pm.unregister(to_unregister)

def test_serve_with_get_and_token():
    if False:
        for i in range(10):
            print('nop')
    runner = CliRunner()
    result1 = runner.invoke(cli, ['create-token', '--secret', 'sekrit', 'root'])
    token = result1.output.strip()
    result2 = runner.invoke(cli, ['serve', '--secret', 'sekrit', '--get', '/-/actor.json', '--token', token])
    assert 0 == result2.exit_code, result2.output
    assert json.loads(result2.output) == {'actor': {'id': 'root', 'token': 'dstok'}}

def test_serve_with_get_exit_code_for_error():
    if False:
        print('Hello World!')
    runner = CliRunner()
    result = runner.invoke(cli, ['serve', '--memory', '--get', '/this-is-404'], catch_exceptions=False)
    assert result.exit_code == 1
    assert '404' in result.output

def test_serve_get_actor():
    if False:
        print('Hello World!')
    runner = CliRunner()
    result = runner.invoke(cli, ['serve', '--memory', '--get', '/-/actor.json', '--actor', '{"id": "root", "extra": "x"}'], catch_exceptions=False)
    assert result.exit_code == 0
    assert json.loads(result.output) == {'actor': {'id': 'root', 'extra': 'x'}}