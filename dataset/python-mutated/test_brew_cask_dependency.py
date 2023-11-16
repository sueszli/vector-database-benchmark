import pytest
from thefuck.rules.brew_cask_dependency import match, get_new_command
from thefuck.types import Command
output = 'sshfs: OsxfuseRequirement unsatisfied!\n\nYou can install with Homebrew-Cask:\n  brew cask install osxfuse\n\nYou can download from:\n  https://osxfuse.github.io/\nError: An unsatisfied requirement failed this build.'

def test_match():
    if False:
        while True:
            i = 10
    command = Command('brew install sshfs', output)
    assert match(command)

@pytest.mark.parametrize('script, output', [('brew link sshfs', output), ('cat output', output), ('brew install sshfs', '')])
def test_not_match(script, output):
    if False:
        while True:
            i = 10
    command = Command(script, output)
    assert not match(command)

@pytest.mark.parametrize('before, after', [('brew install sshfs', 'brew cask install osxfuse && brew install sshfs')])
def test_get_new_command(before, after):
    if False:
        while True:
            i = 10
    command = Command(before, output)
    assert get_new_command(command) == after