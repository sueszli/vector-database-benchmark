import pytest
import shutil
import os
from tests.utils import http
NAKED_BASE_TEMPLATE = "usage:\n    http {extra_args}[METHOD] URL [REQUEST_ITEM ...]\n\nerror:\n    {error_msg}\n\nfor more information:\n    run 'http --help' or visit https://httpie.io/docs/cli\n\n"
NAKED_HELP_MESSAGE = NAKED_BASE_TEMPLATE.format(extra_args='', error_msg='the following arguments are required: URL')
NAKED_HELP_MESSAGE_PRETTY_WITH_NO_ARG = NAKED_BASE_TEMPLATE.format(extra_args='--pretty {all, colors, format, none} ', error_msg='argument --pretty: expected one argument')
NAKED_HELP_MESSAGE_PRETTY_WITH_INVALID_ARG = NAKED_BASE_TEMPLATE.format(extra_args='--pretty {all, colors, format, none} ', error_msg="argument --pretty: invalid choice: '$invalid' (choose from 'all', 'colors', 'format', 'none')")
PREDEFINED_TERMINAL_SIZE = (200, 100)

@pytest.fixture(scope='function')
def ignore_terminal_size(monkeypatch):
    if False:
        return 10
    'Some tests wrap/crop the output depending on the\n    size of the executed terminal, which might not be consistent\n    through all runs.\n\n    This fixture ensures every run uses the same exact configuration.\n    '

    def fake_terminal_size(*args, **kwargs):
        if False:
            return 10
        return os.terminal_size(PREDEFINED_TERMINAL_SIZE)
    monkeypatch.setitem(os.environ, 'COLUMNS', str(PREDEFINED_TERMINAL_SIZE[0]))
    monkeypatch.setattr(shutil, 'get_terminal_size', fake_terminal_size)
    monkeypatch.setattr(os, 'get_terminal_size', fake_terminal_size)

@pytest.mark.parametrize('args, expected_msg', [([], NAKED_HELP_MESSAGE), (['--pretty'], NAKED_HELP_MESSAGE_PRETTY_WITH_NO_ARG), (['pie.dev', '--pretty'], NAKED_HELP_MESSAGE_PRETTY_WITH_NO_ARG), (['--pretty', '$invalid'], NAKED_HELP_MESSAGE_PRETTY_WITH_INVALID_ARG)])
def test_naked_invocation(ignore_terminal_size, args, expected_msg):
    if False:
        return 10
    result = http(*args, tolerate_error_exit_status=True)
    assert result.stderr == expected_msg