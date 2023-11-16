import pytest
import click

def test_command_no_parens(runner):
    if False:
        return 10

    @click.command
    def cli():
        if False:
            while True:
                i = 10
        click.echo('hello')
    result = runner.invoke(cli)
    assert result.exception is None
    assert result.output == 'hello\n'

def test_custom_command_no_parens(runner):
    if False:
        for i in range(10):
            print('nop')

    class CustomCommand(click.Command):
        pass

    class CustomGroup(click.Group):
        command_class = CustomCommand

    @click.group(cls=CustomGroup)
    def grp():
        if False:
            return 10
        pass

    @grp.command
    def cli():
        if False:
            i = 10
            return i + 15
        click.echo('hello custom command class')
    result = runner.invoke(cli)
    assert result.exception is None
    assert result.output == 'hello custom command class\n'

def test_group_no_parens(runner):
    if False:
        print('Hello World!')

    @click.group
    def grp():
        if False:
            i = 10
            return i + 15
        click.echo('grp1')

    @grp.command
    def cmd1():
        if False:
            for i in range(10):
                print('nop')
        click.echo('cmd1')

    @grp.group
    def grp2():
        if False:
            return 10
        click.echo('grp2')

    @grp2.command
    def cmd2():
        if False:
            return 10
        click.echo('cmd2')
    result = runner.invoke(grp, ['cmd1'])
    assert result.exception is None
    assert result.output == 'grp1\ncmd1\n'
    result = runner.invoke(grp, ['grp2', 'cmd2'])
    assert result.exception is None
    assert result.output == 'grp1\ngrp2\ncmd2\n'

def test_params_argument(runner):
    if False:
        for i in range(10):
            print('nop')
    opt = click.Argument(['a'])

    @click.command(params=[opt])
    @click.argument('b')
    def cli(a, b):
        if False:
            while True:
                i = 10
        click.echo(f'{a} {b}')
    assert cli.params[0].name == 'a'
    assert cli.params[1].name == 'b'
    result = runner.invoke(cli, ['1', '2'])
    assert result.output == '1 2\n'

@pytest.mark.parametrize('name', ['init_data', 'init_data_command', 'init_data_cmd', 'init_data_group', 'init_data_grp'])
def test_generate_name(name: str) -> None:
    if False:
        while True:
            i = 10

    def f():
        if False:
            while True:
                i = 10
        pass
    f.__name__ = name
    f = click.command(f)
    assert f.name == 'init-data'