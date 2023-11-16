import pytest
import click.shell_completion
from click.core import Argument
from click.core import Command
from click.core import Group
from click.core import Option
from click.shell_completion import add_completion_class
from click.shell_completion import CompletionItem
from click.shell_completion import ShellComplete
from click.types import Choice
from click.types import File
from click.types import Path

def _get_completions(cli, args, incomplete):
    if False:
        while True:
            i = 10
    comp = ShellComplete(cli, {}, cli.name, '_CLICK_COMPLETE')
    return comp.get_completions(args, incomplete)

def _get_words(cli, args, incomplete):
    if False:
        return 10
    return [c.value for c in _get_completions(cli, args, incomplete)]

def test_command():
    if False:
        print('Hello World!')
    cli = Command('cli', params=[Option(['-t', '--test'])])
    assert _get_words(cli, [], '') == []
    assert _get_words(cli, [], '-') == ['-t', '--test', '--help']
    assert _get_words(cli, [], '--') == ['--test', '--help']
    assert _get_words(cli, [], '--t') == ['--test']
    assert _get_words(cli, ['-t', 'a'], '-') == ['--help']

def test_group():
    if False:
        i = 10
        return i + 15
    cli = Group('cli', params=[Option(['-a'])], commands=[Command('x'), Command('y')])
    assert _get_words(cli, [], '') == ['x', 'y']
    assert _get_words(cli, [], '-') == ['-a', '--help']

def test_group_command_same_option():
    if False:
        i = 10
        return i + 15
    cli = Group('cli', params=[Option(['-a'])], commands=[Command('x', params=[Option(['-a'])])])
    assert _get_words(cli, [], '-') == ['-a', '--help']
    assert _get_words(cli, ['-a', 'a'], '-') == ['--help']
    assert _get_words(cli, ['-a', 'a', 'x'], '-') == ['-a', '--help']
    assert _get_words(cli, ['-a', 'a', 'x', '-a', 'a'], '-') == ['--help']

def test_chained():
    if False:
        print('Hello World!')
    cli = Group('cli', chain=True, commands=[Command('set', params=[Option(['-y'])]), Command('start'), Group('get', commands=[Command('full')])])
    assert _get_words(cli, [], '') == ['get', 'set', 'start']
    assert _get_words(cli, [], 's') == ['set', 'start']
    assert _get_words(cli, ['set', 'start'], '') == ['get']
    assert _get_words(cli, ['get'], '') == ['full', 'set', 'start']
    assert _get_words(cli, ['get', 'full'], '') == ['set', 'start']
    assert _get_words(cli, ['get'], 's') == ['set', 'start']

def test_help_option():
    if False:
        print('Hello World!')
    cli = Group('cli', commands=[Command('with'), Command('no', add_help_option=False)])
    assert _get_words(cli, ['with'], '--') == ['--help']
    assert _get_words(cli, ['no'], '--') == []

def test_argument_order():
    if False:
        print('Hello World!')
    cli = Command('cli', params=[Argument(['plain']), Argument(['c1'], type=Choice(['a1', 'a2', 'b'])), Argument(['c2'], type=Choice(['c1', 'c2', 'd']))])
    assert _get_words(cli, [], '') == []
    assert _get_words(cli, [], 'a') == []
    assert _get_words(cli, ['x'], 'a') == ['a1', 'a2']
    assert _get_words(cli, ['x', 'b'], 'd') == ['d']

def test_argument_default():
    if False:
        while True:
            i = 10
    cli = Command('cli', add_help_option=False, params=[Argument(['a'], type=Choice(['a']), default='a'), Argument(['b'], type=Choice(['b']), default='b')])
    assert _get_words(cli, [], '') == ['a']
    assert _get_words(cli, ['a'], 'b') == ['b']
    assert _get_words(cli, ['x'], 'b') == ['b']

def test_type_choice():
    if False:
        while True:
            i = 10
    cli = Command('cli', params=[Option(['-c'], type=Choice(['a1', 'a2', 'b']))])
    assert _get_words(cli, ['-c'], '') == ['a1', 'a2', 'b']
    assert _get_words(cli, ['-c'], 'a') == ['a1', 'a2']
    assert _get_words(cli, ['-c'], 'a2') == ['a2']

def test_choice_special_characters():
    if False:
        i = 10
        return i + 15
    cli = Command('cli', params=[Option(['-c'], type=Choice(['!1', '!2', '+3']))])
    assert _get_words(cli, ['-c'], '') == ['!1', '!2', '+3']
    assert _get_words(cli, ['-c'], '!') == ['!1', '!2']
    assert _get_words(cli, ['-c'], '!2') == ['!2']

def test_choice_conflicting_prefix():
    if False:
        i = 10
        return i + 15
    cli = Command('cli', params=[Option(['-c'], type=Choice(['!1', '!2', '+3'])), Option(['+p'], is_flag=True)])
    assert _get_words(cli, ['-c'], '') == ['!1', '!2', '+3']
    assert _get_words(cli, ['-c'], '+') == ['+p']

def test_option_count():
    if False:
        return 10
    cli = Command('cli', params=[Option(['-c'], count=True)])
    assert _get_words(cli, ['-c'], '') == []
    assert _get_words(cli, ['-c'], '-') == ['--help']

def test_option_optional():
    if False:
        for i in range(10):
            print('nop')
    cli = Command('cli', add_help_option=False, params=[Option(['--name'], is_flag=False, flag_value='value'), Option(['--flag'], is_flag=True)])
    assert _get_words(cli, ['--name'], '') == []
    assert _get_words(cli, ['--name'], '-') == ['--flag']
    assert _get_words(cli, ['--name', '--flag'], '-') == []

@pytest.mark.parametrize(('type', 'expect'), [(File(), 'file'), (Path(), 'file'), (Path(file_okay=False), 'dir')])
def test_path_types(type, expect):
    if False:
        for i in range(10):
            print('nop')
    cli = Command('cli', params=[Option(['-f'], type=type)])
    out = _get_completions(cli, ['-f'], 'ab')
    assert len(out) == 1
    c = out[0]
    assert c.value == 'ab'
    assert c.type == expect

def test_absolute_path():
    if False:
        while True:
            i = 10
    cli = Command('cli', params=[Option(['-f'], type=Path())])
    out = _get_completions(cli, ['-f'], '/ab')
    assert len(out) == 1
    c = out[0]
    assert c.value == '/ab'

def test_option_flag():
    if False:
        print('Hello World!')
    cli = Command('cli', add_help_option=False, params=[Option(['--on/--off']), Argument(['a'], type=Choice(['a1', 'a2', 'b']))])
    assert _get_words(cli, [], '--') == ['--on', '--off']
    assert _get_words(cli, ['--on'], 'a') == ['a1', 'a2']

def test_option_custom():
    if False:
        while True:
            i = 10

    def custom(ctx, param, incomplete):
        if False:
            while True:
                i = 10
        return [incomplete.upper()]
    cli = Command('cli', params=[Argument(['x']), Argument(['y']), Argument(['z'], shell_complete=custom)])
    assert _get_words(cli, ['a', 'b'], '') == ['']
    assert _get_words(cli, ['a', 'b'], 'c') == ['C']

def test_option_multiple():
    if False:
        for i in range(10):
            print('nop')
    cli = Command('type', params=[Option(['-m'], type=Choice(['a', 'b']), multiple=True), Option(['-f'])])
    assert _get_words(cli, ['-m'], '') == ['a', 'b']
    assert '-m' in _get_words(cli, ['-m', 'a'], '-')
    assert _get_words(cli, ['-m', 'a', '-m'], '') == ['a', 'b']
    assert '-c' not in _get_words(cli, ['-c', 'f'], '-')

def test_option_nargs():
    if False:
        for i in range(10):
            print('nop')
    cli = Command('cli', params=[Option(['-c'], type=Choice(['a', 'b']), nargs=2)])
    assert _get_words(cli, ['-c'], '') == ['a', 'b']
    assert _get_words(cli, ['-c', 'a'], '') == ['a', 'b']
    assert _get_words(cli, ['-c', 'a', 'b'], '') == []

def test_argument_nargs():
    if False:
        while True:
            i = 10
    cli = Command('cli', params=[Argument(['x'], type=Choice(['a', 'b']), nargs=2), Argument(['y'], type=Choice(['c', 'd']), nargs=-1), Option(['-z'])])
    assert _get_words(cli, [], '') == ['a', 'b']
    assert _get_words(cli, ['a'], '') == ['a', 'b']
    assert _get_words(cli, ['a', 'b'], '') == ['c', 'd']
    assert _get_words(cli, ['a', 'b', 'c'], '') == ['c', 'd']
    assert _get_words(cli, ['a', 'b', 'c', 'd'], '') == ['c', 'd']
    assert _get_words(cli, ['a', '-z', '1'], '') == ['a', 'b']
    assert _get_words(cli, ['a', '-z', '1', 'b'], '') == ['c', 'd']

def test_double_dash():
    if False:
        return 10
    cli = Command('cli', add_help_option=False, params=[Option(['--opt']), Argument(['name'], type=Choice(['name', '--', '-o', '--opt']))])
    assert _get_words(cli, [], '-') == ['--opt']
    assert _get_words(cli, ['value'], '-') == ['--opt']
    assert _get_words(cli, [], '') == ['name', '--', '-o', '--opt']
    assert _get_words(cli, ['--'], '') == ['name', '--', '-o', '--opt']

def test_hidden():
    if False:
        while True:
            i = 10
    cli = Group('cli', commands=[Command('hidden', add_help_option=False, hidden=True, params=[Option(['-a']), Option(['-b'], type=Choice(['a', 'b']), hidden=True)])])
    assert 'hidden' not in _get_words(cli, [], '')
    assert 'hidden' not in _get_words(cli, [], 'hidden')
    assert _get_words(cli, ['hidden'], '-') == ['-a']
    assert _get_words(cli, ['hidden', '-b'], '') == ['a', 'b']

def test_add_different_name():
    if False:
        for i in range(10):
            print('nop')
    cli = Group('cli', commands={'renamed': Command('original')})
    words = _get_words(cli, [], '')
    assert 'renamed' in words
    assert 'original' not in words

def test_completion_item_data():
    if False:
        return 10
    c = CompletionItem('test', a=1)
    assert c.a == 1
    assert c.b is None

@pytest.fixture()
def _patch_for_completion(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr('click.shell_completion.BashComplete._check_version', lambda self: True)

@pytest.mark.parametrize('shell', ['bash', 'zsh', 'fish'])
@pytest.mark.usefixtures('_patch_for_completion')
def test_full_source(runner, shell):
    if False:
        return 10
    cli = Group('cli', commands=[Command('a'), Command('b')])
    result = runner.invoke(cli, env={'_CLI_COMPLETE': f'{shell}_source'})
    assert f'_CLI_COMPLETE={shell}_complete' in result.output

@pytest.mark.parametrize(('shell', 'env', 'expect'), [('bash', {'COMP_WORDS': '', 'COMP_CWORD': '0'}, 'plain,a\nplain,b\n'), ('bash', {'COMP_WORDS': 'a b', 'COMP_CWORD': '1'}, 'plain,b\n'), ('zsh', {'COMP_WORDS': '', 'COMP_CWORD': '0'}, 'plain\na\n_\nplain\nb\nbee\n'), ('zsh', {'COMP_WORDS': 'a b', 'COMP_CWORD': '1'}, 'plain\nb\nbee\n'), ('fish', {'COMP_WORDS': '', 'COMP_CWORD': ''}, 'plain,a\nplain,b\tbee\n'), ('fish', {'COMP_WORDS': 'a b', 'COMP_CWORD': 'b'}, 'plain,b\tbee\n')])
@pytest.mark.usefixtures('_patch_for_completion')
def test_full_complete(runner, shell, env, expect):
    if False:
        return 10
    cli = Group('cli', commands=[Command('a'), Command('b', help='bee')])
    env['_CLI_COMPLETE'] = f'{shell}_complete'
    result = runner.invoke(cli, env=env)
    assert result.output == expect

@pytest.mark.usefixtures('_patch_for_completion')
def test_context_settings(runner):
    if False:
        return 10

    def complete(ctx, param, incomplete):
        if False:
            return 10
        return ctx.obj['choices']
    cli = Command('cli', params=[Argument('x', shell_complete=complete)])
    result = runner.invoke(cli, obj={'choices': ['a', 'b']}, env={'COMP_WORDS': '', 'COMP_CWORD': '0', '_CLI_COMPLETE': 'bash_complete'})
    assert result.output == 'plain,a\nplain,b\n'

@pytest.mark.parametrize(('value', 'expect'), [(False, ['Au', 'al']), (True, ['al'])])
def test_choice_case_sensitive(value, expect):
    if False:
        print('Hello World!')
    cli = Command('cli', params=[Option(['-a'], type=Choice(['Au', 'al', 'Bc'], case_sensitive=value))])
    completions = _get_words(cli, ['-a'], 'a')
    assert completions == expect

@pytest.fixture()
def _restore_available_shells(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    prev_available_shells = click.shell_completion._available_shells.copy()
    click.shell_completion._available_shells.clear()
    yield
    click.shell_completion._available_shells.clear()
    click.shell_completion._available_shells.update(prev_available_shells)

@pytest.mark.usefixtures('_restore_available_shells')
def test_add_completion_class():
    if False:
        while True:
            i = 10
    assert 'mysh' not in click.shell_completion._available_shells

    class MyshComplete(ShellComplete):
        name = 'mysh'
        source_template = 'dummy source'
    assert 'mysh' not in click.shell_completion._available_shells
    assert add_completion_class(MyshComplete) is MyshComplete
    assert 'mysh' in click.shell_completion._available_shells
    assert click.shell_completion._available_shells['mysh'] is MyshComplete

@pytest.mark.usefixtures('_restore_available_shells')
def test_add_completion_class_with_name():
    if False:
        i = 10
        return i + 15
    assert 'mysh' not in click.shell_completion._available_shells
    assert 'not_mysh' not in click.shell_completion._available_shells

    class MyshComplete(ShellComplete):
        name = 'not_mysh'
        source_template = 'dummy source'
    assert 'mysh' not in click.shell_completion._available_shells
    assert 'not_mysh' not in click.shell_completion._available_shells
    assert add_completion_class(MyshComplete, name='mysh') is MyshComplete
    assert 'mysh' in click.shell_completion._available_shells
    assert 'not_mysh' not in click.shell_completion._available_shells
    assert click.shell_completion._available_shells['mysh'] is MyshComplete

@pytest.mark.usefixtures('_restore_available_shells')
def test_add_completion_class_decorator():
    if False:
        print('Hello World!')
    assert 'mysh' not in click.shell_completion._available_shells

    @add_completion_class
    class MyshComplete(ShellComplete):
        name = 'mysh'
        source_template = 'dummy source'
    assert 'mysh' in click.shell_completion._available_shells
    assert click.shell_completion._available_shells['mysh'] is MyshComplete