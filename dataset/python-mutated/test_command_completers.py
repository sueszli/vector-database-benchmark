from unittest.mock import Mock
import pytest
from xonsh.completer import Completer
from xonsh.completers.commands import complete_command, complete_skipper
from xonsh.parsers.completion_context import CommandArg, CommandContext, CompletionContext
from xonsh.pytest.tools import ON_WINDOWS, completions_from_result, skip_if_on_windows

@pytest.fixture(autouse=True)
def xs_orig_commands_cache(xession):
    if False:
        print('Hello World!')
    pass

def test_complete_command(completion_context_parse):
    if False:
        return 10
    if ON_WINDOWS:
        command = 'dir.exe'
    else:
        command = 'grep'
    comps = complete_command(completion_context_parse(command[:-1], len(command) - 1).command)
    assert command in set(map(str, comps))

@skip_if_on_windows
def test_skipper_command(completion_context_parse):
    if False:
        for i in range(10):
            print('nop')
    assert 'grep' in completions_from_result(complete_skipper(completion_context_parse('sudo gre', 8)))

@skip_if_on_windows
def test_skipper_arg(completion_context_parse, xession, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(xession.shell.shell, 'completer', Completer(), raising=False)
    bash_completer_mock = Mock()
    monkeypatch.setattr(xession, '_completers', {'bash': bash_completer_mock})
    bash_completer_mock.return_value = {'--count '}
    assert '--count ' in completions_from_result(complete_skipper(completion_context_parse('sudo grep --coun', 16)))
    call_args = bash_completer_mock.call_args[0]
    assert len(call_args) == 1
    context = call_args[0]
    assert isinstance(context, CompletionContext)
    assert context.command == CommandContext(args=(CommandArg('grep'),), arg_index=1, prefix='--coun')