"""
Passing steps.
Often needed in examples.

EXAMPLES:

    Given a step passes
    When  another step passes
    Then  a step passes

    Given ...
    When  ...
    Then  it should pass because "the answer is correct".
"""
from __future__ import absolute_import
from behave import step, then

@step('{word:w} step passes')
def step_passes(context, word):
    if False:
        for i in range(10):
            print('nop')
    '\n    Step that always fails, mostly needed in examples.\n    '
    pass

@then('it should pass because "{reason}"')
def then_it_should_pass_because(context, reason):
    if False:
        for i in range(10):
            print('nop')
    '\n    Self documenting step that indicates some reason.\n    '
    pass