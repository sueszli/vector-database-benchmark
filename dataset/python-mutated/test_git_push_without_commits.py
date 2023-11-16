from thefuck.types import Command
from thefuck.rules.git_push_without_commits import get_new_command, match

def test_match():
    if False:
        while True:
            i = 10
    script = 'git push -u origin master'
    output = 'error: src refspec master does not match any\nerror: failed to...'
    assert match(Command(script, output))

def test_not_match():
    if False:
        i = 10
        return i + 15
    script = 'git push -u origin master'
    assert not match(Command(script, 'Everything up-to-date'))

def test_get_new_command():
    if False:
        return 10
    script = 'git push -u origin master'
    output = 'error: src refspec master does not match any\nerror: failed to...'
    new_command = 'git commit -m "Initial commit" && git push -u origin master'
    assert get_new_command(Command(script, output)) == new_command