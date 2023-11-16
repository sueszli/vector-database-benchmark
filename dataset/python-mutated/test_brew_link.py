import pytest
from thefuck.types import Command
from thefuck.rules.brew_link import get_new_command, match

@pytest.fixture
def output():
    if False:
        while True:
            i = 10
    return "Error: Could not symlink bin/gcp\nTarget /usr/local/bin/gcp\nalready exists. You may want to remove it:\n  rm '/usr/local/bin/gcp'\n\nTo force the link and overwrite all conflicting files:\n  brew link --overwrite coreutils\n\nTo list all files that would be deleted:\n  brew link --overwrite --dry-run coreutils\n"

@pytest.fixture
def new_command(formula):
    if False:
        i = 10
        return i + 15
    return 'brew link --overwrite --dry-run {}'.format(formula)

@pytest.mark.parametrize('script', ['brew link coreutils', 'brew ln coreutils'])
def test_match(output, script):
    if False:
        i = 10
        return i + 15
    assert match(Command(script, output))

@pytest.mark.parametrize('script', ['brew link coreutils'])
def test_not_match(script):
    if False:
        return 10
    assert not match(Command(script, ''))

@pytest.mark.parametrize('script, formula, ', [('brew link coreutils', 'coreutils')])
def test_get_new_command(output, new_command, script, formula):
    if False:
        while True:
            i = 10
    assert get_new_command(Command(script, output)) == new_command