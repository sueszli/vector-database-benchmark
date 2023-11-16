import pytest
from thefuck.rules.lein_not_task import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def is_not_task():
    if False:
        i = 10
        return i + 15
    return "'rpl' is not a task. See 'lein help'.\n\nDid you mean this?\n         repl\n         jar\n"

def test_match(is_not_task):
    if False:
        return 10
    assert match(Command('lein rpl', is_not_task))
    assert not match(Command('ls', is_not_task))

def test_get_new_command(is_not_task):
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('lein rpl --help', is_not_task)) == ['lein repl --help', 'lein jar --help']