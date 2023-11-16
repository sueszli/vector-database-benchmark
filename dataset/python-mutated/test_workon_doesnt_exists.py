import pytest
from thefuck.rules.workon_doesnt_exists import match, get_new_command
from thefuck.types import Command

@pytest.fixture(autouse=True)
def envs(mocker):
    if False:
        return 10
    return mocker.patch('thefuck.rules.workon_doesnt_exists._get_all_environments', return_value=['thefuck', 'code_view'])

@pytest.mark.parametrize('script', ['workon tehfuck', 'workon code-view', 'workon new-env'])
def test_match(script):
    if False:
        while True:
            i = 10
    assert match(Command(script, ''))

@pytest.mark.parametrize('script', ['workon thefuck', 'workon code_view', 'work on tehfuck'])
def test_not_match(script):
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script, result', [('workon tehfuck', 'workon thefuck'), ('workon code-view', 'workon code_view'), ('workon zzzz', 'mkvirtualenv zzzz')])
def test_get_new_command(script, result):
    if False:
        print('Hello World!')
    assert get_new_command(Command(script, ''))[0] == result