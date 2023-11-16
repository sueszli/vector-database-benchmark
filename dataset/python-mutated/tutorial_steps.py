"""Step implementations for tutorial example."""
from behave import *

@given('we have behave installed')
def step_impl(context):
    if False:
        while True:
            i = 10
    pass

@when('we implement a test')
def step_impl(context):
    if False:
        for i in range(10):
            print('nop')
    assert True is not False

@then('behave will test it for us!')
def step_impl(context):
    if False:
        i = 10
        return i + 15
    assert context.failed is False