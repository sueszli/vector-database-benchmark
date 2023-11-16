"""
Provides step definitions for behave based on behave4cmd.

REQUIRES:
  * behave4cmd.steplib.output steps (command output from behave).
"""
from __future__ import absolute_import
from behave import then
from behave.runner_util import make_undefined_step_snippet

def text_indent(text, indent_size=0):
    if False:
        i = 10
        return i + 15
    prefix = ' ' * indent_size
    return prefix.join(text.splitlines(True))

@then(u'an undefined-step snippets section exists')
def step_undefined_step_snippets_section_exists(context):
    if False:
        return 10
    '\n    Checks if an undefined-step snippet section is in behave command output.\n    '
    context.execute_steps(u'\n        Then the command output should contain:\n            """\n            You can implement step definitions for undefined steps with these snippets:\n            """\n    ')

@then(u'an undefined-step snippet should exist for "{step}"')
def step_undefined_step_snippet_should_exist_for(context, step):
    if False:
        print('Hello World!')
    '\n    Checks if an undefined-step snippet is provided for a step\n    in behave command output (last command).\n\n    EXAMPLE:\n        Then an undefined-step snippet should exist for "Given an undefined step"\n    '
    undefined_step_snippet = make_undefined_step_snippet(step)
    context.execute_steps(u'Then the command output should contain:\n    """\n    {undefined_step_snippet}\n    """\n    '.format(undefined_step_snippet=text_indent(undefined_step_snippet, 4)))

@then(u'an undefined-step snippet should not exist for "{step}"')
def step_undefined_step_snippet_should_not_exist_for(context, step):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks if an undefined-step snippet is provided for a step\n    in behave command output (last command).\n    '
    undefined_step_snippet = make_undefined_step_snippet(step)
    context.execute_steps(u'Then the command output should not contain:\n    """\n    {undefined_step_snippet}\n    """\n    '.format(undefined_step_snippet=text_indent(undefined_step_snippet, 4)))

@then(u'undefined-step snippets should exist for')
def step_undefined_step_snippets_should_exist_for_table(context):
    if False:
        while True:
            i = 10
    '\n    Checks if undefined-step snippets are provided.\n\n    EXAMPLE:\n        Then undefined-step snippets should exist for:\n            | Step |\n            | When an undefined step is used |\n            | Then another undefined step is used |\n    '
    assert context.table, 'REQUIRES: table'
    for row in context.table.rows:
        step = row['Step']
        step_undefined_step_snippet_should_exist_for(context, step)

@then(u'undefined-step snippets should not exist for')
def step_undefined_step_snippets_should_not_exist_for_table(context):
    if False:
        return 10
    '\n    Checks if undefined-step snippets are not provided.\n\n    EXAMPLE:\n        Then undefined-step snippets should not exist for:\n            | Step |\n            | When an known step is used |\n            | Then another known step is used |\n    '
    assert context.table, 'REQUIRES: table'
    for row in context.table.rows:
        step = row['Step']
        step_undefined_step_snippet_should_not_exist_for(context, step)