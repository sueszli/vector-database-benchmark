import pytest
from io import BytesIO
from thefuck.types import Command
from thefuck.rules.gulp_not_task import match, get_new_command

def output(task):
    if False:
        i = 10
        return i + 15
    return "[00:41:11] Using gulpfile gulpfile.js\n[00:41:11] Task '{}' is not in your gulpfile\n[00:41:11] Please check the documentation for proper gulpfile formatting\n".format(task)

def test_match():
    if False:
        for i in range(10):
            print('nop')
    assert match(Command('gulp srve', output('srve')))

@pytest.mark.parametrize('script, stdout', [('gulp serve', ''), ('cat srve', output('srve'))])
def test_not_march(script, stdout):
    if False:
        while True:
            i = 10
    assert not match(Command(script, stdout))

def test_get_new_command(mocker):
    if False:
        i = 10
        return i + 15
    mock = mocker.patch('subprocess.Popen')
    mock.return_value.stdout = BytesIO(b'serve \nbuild \ndefault \n')
    command = Command('gulp srve', output('srve'))
    assert get_new_command(command) == ['gulp serve', 'gulp default']