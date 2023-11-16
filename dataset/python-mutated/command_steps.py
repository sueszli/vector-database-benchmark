"""
Provides step definitions to:

    * run commands, like behave
    * create textual files within a working directory

TODO:
  matcher that ignores empty lines and whitespace and has contains comparison
"""
from __future__ import absolute_import, print_function
from behave import when, then, matchers
from behave4cmd0 import command_shell, command_util, textutil
from behave4cmd0.step_util import DEBUG, on_assert_failed_print_details, normalize_text_with_placeholders
from hamcrest import assert_that, equal_to, is_not
matchers.register_type(int=int)

@when(u'I run "{command}"')
@when(u'I run `{command}`')
def step_i_run_command(context, command):
    if False:
        print('Hello World!')
    '\n    Run a command as subprocess, collect its output and returncode.\n    '
    command_util.ensure_workdir_exists(context)
    context.command_result = command_shell.run(command, cwd=context.workdir)
    command_util.workdir_save_coverage_files(context.workdir)
    if False and DEBUG:
        print(u'run_command: {0}'.format(command))
        print(u'run_command.output {0}'.format(context.command_result.output))

@when(u'I successfully run "{command}"')
@when(u'I successfully run `{command}`')
def step_i_successfully_run_command(context, command):
    if False:
        i = 10
        return i + 15
    step_i_run_command(context, command)
    step_it_should_pass(context)

@then(u'it should fail with result "{result:int}"')
def step_it_should_fail_with_result(context, result):
    if False:
        while True:
            i = 10
    assert_that(context.command_result.returncode, equal_to(result))
    assert_that(result, is_not(equal_to(0)))

@then(u'the command should fail with returncode="{result:int}"')
def step_it_should_fail_with_returncode(context, result):
    if False:
        return 10
    assert_that(context.command_result.returncode, equal_to(result))
    assert_that(result, is_not(equal_to(0)))

@then(u'the command returncode is "{result:int}"')
def step_the_command_returncode_is(context, result):
    if False:
        print('Hello World!')
    assert_that(context.command_result.returncode, equal_to(result))

@then(u'the command returncode is non-zero')
def step_the_command_returncode_is_nonzero(context):
    if False:
        i = 10
        return i + 15
    assert_that(context.command_result.returncode, is_not(equal_to(0)))

@then(u'it should pass')
def step_it_should_pass(context):
    if False:
        print('Hello World!')
    assert_that(context.command_result.returncode, equal_to(0), context.command_result.output)

@then(u'it should fail')
def step_it_should_fail(context):
    if False:
        print('Hello World!')
    assert_that(context.command_result.returncode, is_not(equal_to(0)), context.command_result.output)

@then(u'it should pass with')
def step_it_should_pass_with(context):
    if False:
        while True:
            i = 10
    '\n    EXAMPLE:\n        ...\n        when I run "behave ..."\n        then it should pass with:\n            """\n            TEXT\n            """\n    '
    assert context.text is not None, 'ENSURE: multiline text is provided.'
    step_command_output_should_contain(context)
    assert_that(context.command_result.returncode, equal_to(0), context.command_result.output)

@then(u'it should fail with')
def step_it_should_fail_with(context):
    if False:
        return 10
    '\n    EXAMPLE:\n        ...\n        when I run "behave ..."\n        then it should fail with:\n            """\n            TEXT\n            """\n    '
    assert context.text is not None, 'ENSURE: multiline text is provided.'
    step_command_output_should_contain(context)
    assert_that(context.command_result.returncode, is_not(equal_to(0)))

@then(u'the command output should contain "{text}"')
def step_command_output_should_contain_text(context, text):
    if False:
        i = 10
        return i + 15
    '\n    EXAMPLE:\n        ...\n        Then the command output should contain "TEXT"\n    '
    expected_text = normalize_text_with_placeholders(context, text)
    actual_output = context.command_result.output
    with on_assert_failed_print_details(actual_output, expected_text):
        textutil.assert_normtext_should_contain(actual_output, expected_text)

@then(u'the command output should not contain "{text}"')
def step_command_output_should_not_contain_text(context, text):
    if False:
        i = 10
        return i + 15
    '\n    EXAMPLE:\n        ...\n        then the command output should not contain "TEXT"\n    '
    expected_text = normalize_text_with_placeholders(context, text)
    actual_output = context.command_result.output
    with on_assert_failed_print_details(actual_output, expected_text):
        textutil.assert_normtext_should_not_contain(actual_output, expected_text)

@then(u'the command output should contain "{text}" {count:d} times')
def step_command_output_should_contain_text_multiple_times(context, text, count):
    if False:
        return 10
    '\n    EXAMPLE:\n        ...\n        Then the command output should contain "TEXT" 3 times\n    '
    assert count >= 0
    expected_text = normalize_text_with_placeholders(context, text)
    actual_output = context.command_result.output
    expected_text_part = expected_text
    with on_assert_failed_print_details(actual_output, expected_text_part):
        textutil.assert_normtext_should_contain_multiple_times(actual_output, expected_text_part, count)

@then(u'the command output should contain exactly "{text}"')
def step_command_output_should_contain_exactly_text(context, text):
    if False:
        i = 10
        return i + 15
    '\n    Verifies that the command output of the last command contains the\n    expected text.\n\n    .. code-block:: gherkin\n\n        When I run "echo Hello"\n        Then the command output should contain "Hello"\n    '
    expected_text = normalize_text_with_placeholders(context, text)
    actual_output = context.command_result.output
    textutil.assert_text_should_contain_exactly(actual_output, expected_text)

@then(u'the command output should not contain exactly "{text}"')
def step_command_output_should_not_contain_exactly_text(context, text):
    if False:
        return 10
    expected_text = normalize_text_with_placeholders(context, text)
    actual_output = context.command_result.output
    textutil.assert_text_should_not_contain_exactly(actual_output, expected_text)

@then(u'the command output should contain')
def step_command_output_should_contain(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    EXAMPLE:\n        ...\n        when I run "behave ..."\n        then it should pass\n        and  the command output should contain:\n            """\n            TEXT\n            """\n    '
    assert context.text is not None, 'REQUIRE: multi-line text'
    step_command_output_should_contain_text(context, context.text)

@then(u'the command output should not contain')
def step_command_output_should_not_contain(context):
    if False:
        print('Hello World!')
    '\n    EXAMPLE:\n        ...\n        when I run "behave ..."\n        then it should pass\n        and  the command output should not contain:\n            """\n            TEXT\n            """\n    '
    assert context.text is not None, 'REQUIRE: multi-line text'
    text = context.text.rstrip()
    step_command_output_should_not_contain_text(context, text)

@then(u'the command output should contain {count:d} times')
def step_command_output_should_contain_multiple_times(context, count):
    if False:
        while True:
            i = 10
    '\n    EXAMPLE:\n        ...\n        when I run "behave ..."\n        then it should pass\n        and  the command output should contain 2 times:\n            """\n            TEXT\n            """\n    '
    assert context.text is not None, 'REQUIRE: multi-line text'
    text = context.text.rstrip()
    step_command_output_should_contain_text_multiple_times(context, text, count)

@then(u'the command output should contain exactly')
def step_command_output_should_contain_exactly_with_multiline_text(context):
    if False:
        while True:
            i = 10
    assert context.text is not None, 'REQUIRE: multi-line text'
    text = context.text.rstrip()
    step_command_output_should_contain_exactly_text(context, text)

@then(u'the command output should not contain exactly')
def step_command_output_should_contain_not_exactly_with_multiline_text(context):
    if False:
        print('Hello World!')
    assert context.text is not None, 'REQUIRE: multi-line text'
    text = context.text.rstrip()
    step_command_output_should_not_contain_exactly_text(context, text)

@then(u'the command output should match /{pattern}/')
@then(u'the command output should match "{pattern}"')
def step_command_output_should_match_pattern(context, pattern):
    if False:
        return 10
    'Verifies that command output matches the ``pattern``.\n\n    :param pattern: Regular expression pattern to use (as string or compiled).\n\n    .. code-block:: gherkin\n\n        # -- STEP-SCHEMA: Then the command output should match /{pattern}/\n        Scenario:\n          When I run `echo Hello world`\n          Then the command output should match /Hello \\w+/\n    '
    text = context.command_result.output.strip()
    textutil.assert_text_should_match_pattern(text, pattern)

@then(u'the command output should not match /{pattern}/')
@then(u'the command output should not match "{pattern}"')
def step_command_output_should_not_match_pattern(context, pattern):
    if False:
        i = 10
        return i + 15
    text = context.command_result.output
    textutil.assert_text_should_not_match_pattern(text, pattern)

@then(u'the command output should match')
def step_command_output_should_match_with_multiline_text(context):
    if False:
        print('Hello World!')
    assert context.text is not None, 'ENSURE: multiline text is provided.'
    pattern = context.text
    step_command_output_should_match_pattern(context, pattern)

@then(u'the command output should not match')
def step_command_output_should_not_match_with_multiline_text(context):
    if False:
        while True:
            i = 10
    assert context.text is not None, 'ENSURE: multiline text is provided.'
    pattern = context.text
    step_command_output_should_not_match_pattern(context, pattern)