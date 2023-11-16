import pytest
from thefuck.rules.git_lfs_mistype import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def mistype_response():
    if False:
        for i in range(10):
            print('nop')
    return '\nError: unknown command "evn" for "git-lfs"\n\nDid you mean this?\n        env\n        ext\n\nRun \'git-lfs --help\' for usage.\n    '

def test_match(mistype_response):
    if False:
        return 10
    assert match(Command('git lfs evn', mistype_response))
    err_response = 'bash: git: command not found'
    assert not match(Command('git lfs env', err_response))
    assert not match(Command('docker lfs env', mistype_response))

def test_get_new_command(mistype_response):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('git lfs evn', mistype_response)) == ['git lfs env', 'git lfs ext']