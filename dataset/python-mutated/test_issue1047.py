"""
https://github.com/behave/behave/issues/1047
"""
from __future__ import absolute_import, print_function
from behave.parser import parse_steps

def test_issue_1047_step_type_for_generic_steps_is_inherited():
    if False:
        while True:
            i = 10
    'Verifies that issue #1047 is fixed.'
    text = u'When my step\nAnd my second step\n* my third step\n'
    steps = parse_steps(text)
    assert steps[-1].step_type == 'when'

def test_issue_1047_step_type_if_only_generic_steps_are_used():
    if False:
        return 10
    text = u'* my step\n* my another step\n'
    steps = parse_steps(text)
    assert steps[0].step_type == 'given'
    assert steps[1].step_type == 'given'