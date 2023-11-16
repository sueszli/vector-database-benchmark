"""
Behave steps for environment variables (process environment).
"""
from __future__ import absolute_import, print_function
import os
from behave import given, when, then, step
from hamcrest import assert_that, is_, is_not

@step(u'I set the environment variable "{env_name}" to "{env_value}"')
def step_I_set_the_environment_variable_to(context, env_name, env_value):
    if False:
        while True:
            i = 10
    if not hasattr(context, 'environ'):
        context.environ = {}
    context.environ[env_name] = env_value
    os.environ[env_name] = env_value

@step(u'I remove the environment variable "{env_name}"')
def step_I_remove_the_environment_variable(context, env_name):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(context, 'environ'):
        context.environ = {}
    context.environ[env_name] = ''
    os.environ[env_name] = ''
    del context.environ[env_name]
    del os.environ[env_name]

@given(u'the environment variable "{env_name}" exists')
@then(u'the environment variable "{env_name}" exists')
def step_the_environment_variable_exists(context, env_name):
    if False:
        i = 10
        return i + 15
    env_variable_value = os.environ.get(env_name)
    assert_that(env_variable_value, is_not(None))

@given(u'the environment variable "{env_name}" does not exist')
@then(u'the environment variable "{env_name}" does not exist')
def step_I_set_the_environment_variable_to(context, env_name):
    if False:
        while True:
            i = 10
    env_variable_value = os.environ.get(env_name)
    assert_that(env_variable_value, is_(None))