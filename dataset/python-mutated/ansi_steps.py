from __future__ import absolute_import
from behave import then
from behave4cmd0.command_steps import step_command_output_should_contain_text, step_command_output_should_not_contain_text
CSI = u'\x1b['

@then(u'the command output should contain ANSI escape sequences')
def step_command_ouput_should_contain_ansi_sequences(context):
    if False:
        print('Hello World!')
    step_command_output_should_contain_text(context, CSI)

@then(u'the command output should not contain any ANSI escape sequences')
def step_command_ouput_should_not_contain_ansi_sequences(context):
    if False:
        i = 10
        return i + 15
    step_command_output_should_not_contain_text(context, CSI)