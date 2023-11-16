from textwrap import dedent
from behave import then, when
import wrappers
from utils import parse_cli_args_to_dict

@when('we run dbcli with {arg}')
def step_run_cli_with_arg(context, arg):
    if False:
        i = 10
        return i + 15
    wrappers.run_cli(context, run_args=parse_cli_args_to_dict(arg))

@when('we execute a small query')
def step_execute_small_query(context):
    if False:
        return 10
    context.cli.sendline('select 1')

@when('we execute a large query')
def step_execute_large_query(context):
    if False:
        print('Hello World!')
    context.cli.sendline('select {}'.format(','.join([str(n) for n in range(1, 50)])))

@then('we see small results in horizontal format')
def step_see_small_results(context):
    if False:
        i = 10
        return i + 15
    wrappers.expect_pager(context, dedent('        +---+\r\n        | 1 |\r\n        +---+\r\n        | 1 |\r\n        +---+\r\n        \r\n        '), timeout=5)
    wrappers.expect_exact(context, '1 row in set', timeout=2)

@then('we see large results in vertical format')
def step_see_large_results(context):
    if False:
        return 10
    rows = ['{n:3}| {n}'.format(n=str(n)) for n in range(1, 50)]
    expected = '***************************[ 1. row ]***************************\r\n' + '{}\r\n'.format('\r\n'.join(rows) + '\r\n')
    wrappers.expect_pager(context, expected, timeout=10)
    wrappers.expect_exact(context, '1 row in set', timeout=2)