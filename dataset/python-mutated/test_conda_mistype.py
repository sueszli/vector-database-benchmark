import pytest
from thefuck.rules.conda_mistype import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def mistype_response():
    if False:
        print('Hello World!')
    return "\n\nCommandNotFoundError: No command 'conda lst'.\nDid you mean 'conda list'?\n\n    "

def test_match(mistype_response):
    if False:
        i = 10
        return i + 15
    assert match(Command('conda lst', mistype_response))
    err_response = 'bash: codna: command not found'
    assert not match(Command('codna list', err_response))

def test_get_new_command(mistype_response):
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('conda lst', mistype_response)) == ['conda list']