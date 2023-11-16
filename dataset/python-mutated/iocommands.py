import os
import wrappers
from behave import when, then
from textwrap import dedent

@when('we start external editor providing a file name')
def step_edit_file(context):
    if False:
        while True:
            i = 10
    'Edit file with external editor.'
    context.editor_file_name = os.path.join(context.package_root, 'test_file_{0}.sql'.format(context.conf['vi']))
    if os.path.exists(context.editor_file_name):
        os.remove(context.editor_file_name)
    context.cli.sendline('\\e {0}'.format(os.path.basename(context.editor_file_name)))
    wrappers.expect_exact(context, 'Entering Ex mode.  Type "visual" to go to Normal mode.', timeout=2)
    wrappers.expect_exact(context, '\r\n:', timeout=2)

@when('we type "{query}" in the editor')
def step_edit_type_sql(context, query):
    if False:
        print('Hello World!')
    context.cli.sendline('i')
    context.cli.sendline(query)
    context.cli.sendline('.')
    wrappers.expect_exact(context, '\r\n:', timeout=2)

@when('we exit the editor')
def step_edit_quit(context):
    if False:
        while True:
            i = 10
    context.cli.sendline('x')
    wrappers.expect_exact(context, 'written', timeout=2)

@then('we see "{query}" in prompt')
def step_edit_done_sql(context, query):
    if False:
        return 10
    for match in query.split(' '):
        wrappers.expect_exact(context, match, timeout=5)
    context.cli.sendcontrol('c')
    if context.editor_file_name and os.path.exists(context.editor_file_name):
        os.remove(context.editor_file_name)

@when(u'we tee output')
def step_tee_ouptut(context):
    if False:
        return 10
    context.tee_file_name = os.path.join(context.package_root, 'tee_file_{0}.sql'.format(context.conf['vi']))
    if os.path.exists(context.tee_file_name):
        os.remove(context.tee_file_name)
    context.cli.sendline('tee {0}'.format(os.path.basename(context.tee_file_name)))

@when(u'we select "select {param}"')
def step_query_select_number(context, param):
    if False:
        print('Hello World!')
    context.cli.sendline(u'select {}'.format(param))
    wrappers.expect_pager(context, dedent(u'        +{dashes}+\r\n        | {param} |\r\n        +{dashes}+\r\n        | {param} |\r\n        +{dashes}+\r\n        \r\n        '.format(param=param, dashes='-' * (len(param) + 2))), timeout=5)
    wrappers.expect_exact(context, '1 row in set', timeout=2)

@then(u'we see result "{result}"')
def step_see_result(context, result):
    if False:
        i = 10
        return i + 15
    wrappers.expect_exact(context, u'| {} |'.format(result), timeout=2)

@when(u'we query "{query}"')
def step_query(context, query):
    if False:
        i = 10
        return i + 15
    context.cli.sendline(query)

@when(u'we notee output')
def step_notee_output(context):
    if False:
        while True:
            i = 10
    context.cli.sendline('notee')

@then(u'we see 123456 in tee output')
def step_see_123456_in_ouput(context):
    if False:
        print('Hello World!')
    with open(context.tee_file_name) as f:
        assert '123456' in f.read()
    if os.path.exists(context.tee_file_name):
        os.remove(context.tee_file_name)

@then(u'delimiter is set to "{delimiter}"')
def delimiter_is_set(context, delimiter):
    if False:
        print('Hello World!')
    wrappers.expect_exact(context, u'Changed delimiter to {}'.format(delimiter), timeout=2)