from __future__ import print_function
from behave import given

@given(u'a step with table data')
def step_with_table_data(ctx):
    if False:
        i = 10
        return i + 15
    assert ctx.table is not None, 'REQUIRE: step.table'

@given(u'a step with name="{name}"')
def step_with_table_data(ctx, name):
    if False:
        print('Hello World!')
    print(u'name: {}'.format(name))