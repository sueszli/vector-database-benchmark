import pytest
from thefuck.rules.git_push_pull import match, get_new_command
from thefuck.types import Command
git_err = "\nTo /tmp/foo\n ! [rejected]        master -> master (non-fast-forward)\n error: failed to push some refs to '/tmp/bar'\n hint: Updates were rejected because the tip of your current branch is behind\n hint: its remote counterpart. Integrate the remote changes (e.g.\n hint: 'git pull ...') before pushing again.\n hint: See the 'Note about fast-forwards' in 'git push --help' for details.\n"
git_err2 = "\nTo /tmp/foo\n ! [rejected]        master -> master (non-fast-forward)\n error: failed to push some refs to '/tmp/bar'\nhint: Updates were rejected because the remote contains work that you do\nhint: not have locally. This is usually caused by another repository pushing\nhint: to the same ref. You may want to first integrate the remote changes\nhint: (e.g., 'git pull ...') before pushing again.\nhint: See the 'Note about fast-forwards' in 'git push --help' for details.\n"
git_uptodate = 'Everything up-to-date'
git_ok = '\nCounting objects: 3, done.\nDelta compression using up to 4 threads.\nCompressing objects: 100% (2/2), done.\nWriting objects: 100% (3/3), 282 bytes | 0 bytes/s, done.\nTotal 3 (delta 0), reused 0 (delta 0)\nTo /tmp/bar\n   514eed3..f269c79  master -> master\n'

@pytest.mark.parametrize('command', [Command('git push', git_err), Command('git push nvbn', git_err), Command('git push nvbn master', git_err), Command('git push', git_err2), Command('git push nvbn', git_err2), Command('git push nvbn master', git_err2)])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command', [Command('git push', git_ok), Command('git push', git_uptodate), Command('git push nvbn', git_ok), Command('git push nvbn master', git_uptodate), Command('git push nvbn', git_ok), Command('git push nvbn master', git_uptodate)])
def test_not_match(command):
    if False:
        print('Hello World!')
    assert not match(command)

@pytest.mark.parametrize('command, output', [(Command('git push', git_err), 'git pull && git push'), (Command('git push nvbn', git_err), 'git pull nvbn && git push nvbn'), (Command('git push nvbn master', git_err), 'git pull nvbn master && git push nvbn master'), (Command('git push', git_err2), 'git pull && git push'), (Command('git push nvbn', git_err2), 'git pull nvbn && git push nvbn'), (Command('git push nvbn master', git_err2), 'git pull nvbn master && git push nvbn master')])
def test_get_new_command(command, output):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == output