import pytest
from xonsh.completers.base import complete_base
from xonsh.parsers.completion_context import CommandContext, CompletionContext
from xonsh.pytest.tools import ON_WINDOWS
CUR_DIR = '.' if ON_WINDOWS else './'

@pytest.fixture(autouse=True)
def setup(xession, xonsh_execer, monkeypatch, mock_executables_in):
    if False:
        for i in range(10):
            print('nop')
    xession.env['COMMANDS_CACHE_SAVE_INTERMEDIATE'] = False
    xession.env['COMPLETION_QUERY_LIMIT'] = 2000
    mock_executables_in(['cool'])

def test_empty_line(check_completer):
    if False:
        for i in range(10):
            print('nop')
    completions = check_completer('')
    assert completions
    assert completions.issuperset({'cool', 'abs'})
    for exp in ['cool', 'abs']:
        assert exp in completions

def test_empty_subexpr():
    if False:
        for i in range(10):
            print('nop')
    completions = complete_base(CompletionContext(command=CommandContext((), 0, subcmd_opening='$('), python=None))
    completions = set(map(str, completions))
    assert completions
    assert completions.issuperset({'cool'})
    assert 'abs' not in completions