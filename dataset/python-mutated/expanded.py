"""Steps for behavioral style tests are defined in this module.

Each step is defined by the string decorating it. This string is used
to call the step in "*.feature" file.

"""
from behave import when, then
from textwrap import dedent
import wrappers

@when('we prepare the test data')
def step_prepare_data(context):
    if False:
        print('Hello World!')
    'Create table, insert a record.'
    context.cli.sendline('drop table if exists a;')
    wrappers.expect_exact(context, "You're about to run a destructive command.\r\nDo you want to proceed? [y/N]:", timeout=2)
    context.cli.sendline('y')
    wrappers.wait_prompt(context)
    context.cli.sendline('create table a(x integer, y real, z numeric(10, 4));')
    wrappers.expect_pager(context, 'CREATE TABLE\r\n', timeout=2)
    context.cli.sendline('insert into a(x, y, z) values(1, 1.0, 1.0);')
    wrappers.expect_pager(context, 'INSERT 0 1\r\n', timeout=2)

@when('we set expanded {mode}')
def step_set_expanded(context, mode):
    if False:
        i = 10
        return i + 15
    'Set expanded to mode.'
    context.cli.sendline('\\' + f'x {mode}')
    wrappers.expect_exact(context, 'Expanded display is', timeout=2)
    wrappers.wait_prompt(context)

@then('we see {which} data selected')
def step_see_data(context, which):
    if False:
        return 10
    'Select data from expanded test table.'
    if which == 'expanded':
        wrappers.expect_pager(context, dedent('                -[ RECORD 1 ]-------------------------\r\n                x | 1\r\n                y | 1.0\r\n                z | 1.0000\r\n                SELECT 1\r\n            '), timeout=1)
    else:
        wrappers.expect_pager(context, dedent('                +---+-----+--------+\r\n                | x | y   | z      |\r\n                |---+-----+--------|\r\n                | 1 | 1.0 | 1.0000 |\r\n                +---+-----+--------+\r\n                SELECT 1\r\n            '), timeout=1)