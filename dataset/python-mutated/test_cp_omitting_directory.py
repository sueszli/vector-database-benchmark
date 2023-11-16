import pytest
from thefuck.rules.cp_omitting_directory import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('script, output', [('cp dir', 'cp: dor: is a directory'), ('cp dir', "cp: omitting directory 'dir'")])
def test_match(script, output):
    if False:
        return 10
    assert match(Command(script, output))

@pytest.mark.parametrize('script, output', [('some dir', 'cp: dor: is a directory'), ('some dir', "cp: omitting directory 'dir'"), ('cp dir', '')])
def test_not_match(script, output):
    if False:
        for i in range(10):
            print('nop')
    assert not match(Command(script, output))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('cp dir', '')) == 'cp -a dir'