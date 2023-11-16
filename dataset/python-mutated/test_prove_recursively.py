import pytest
from thefuck.rules.prove_recursively import match, get_new_command
from thefuck.types import Command
output = 'Files=0, Tests=0,  0 wallclock secs ( 0.00 usr +  0.00 sys =  0.00 CPU)\nResult: NOTESTS'

@pytest.fixture
def isdir(mocker):
    if False:
        i = 10
        return i + 15
    return mocker.patch('thefuck.rules.prove_recursively.os.path.isdir')

@pytest.mark.parametrize('script, output', [('prove -lv t', output), ('prove app/t', output)])
def test_match(isdir, script, output):
    if False:
        i = 10
        return i + 15
    isdir.return_value = True
    command = Command(script, output)
    assert match(command)

@pytest.mark.parametrize('script, output, isdir_result', [('prove -lv t', output, False), ('prove -r t', output, True), ('prove --recurse t', output, True)])
def test_not_match(isdir, script, output, isdir_result):
    if False:
        print('Hello World!')
    isdir.return_value = isdir_result
    command = Command(script, output)
    assert not match(command)

@pytest.mark.parametrize('before, after', [('prove -lv t', 'prove -r -lv t'), ('prove t', 'prove -r t')])
def test_get_new_command(before, after):
    if False:
        print('Hello World!')
    command = Command(before, output)
    assert get_new_command(command) == after