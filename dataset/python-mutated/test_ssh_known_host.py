import os
import pytest
from thefuck.rules.ssh_known_hosts import match, get_new_command, side_effect
from thefuck.types import Command

@pytest.fixture
def ssh_error(tmpdir):
    if False:
        while True:
            i = 10
    path = os.path.join(str(tmpdir), 'known_hosts')

    def reset(path):
        if False:
            print('Hello World!')
        with open(path, 'w') as fh:
            lines = ['123.234.567.890 asdjkasjdakjsd\n98.765.432.321 ejioweojwejrosj\n111.222.333.444 qwepoiwqepoiss\n']
            fh.writelines(lines)

    def known_hosts(path):
        if False:
            print('Hello World!')
        with open(path, 'r') as fh:
            return fh.readlines()
    reset(path)
    errormsg = u'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\nIT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!\nSomeone could be eavesdropping on you right now (man-in-the-middle attack)!\nIt is also possible that a host key has just been changed.\nThe fingerprint for the RSA key sent by the remote host is\nb6:cb:07:34:c0:a0:94:d3:0d:69:83:31:f4:c5:20:9b.\nPlease contact your system administrator.\nAdd correct host key in {0} to get rid of this message.\nOffending RSA key in {0}:2\nRSA host key for {1} has changed and you have requested strict checking.\nHost key verification failed.'.format(path, '98.765.432.321')
    return (errormsg, path, reset, known_hosts)

def test_match(ssh_error):
    if False:
        i = 10
        return i + 15
    (errormsg, _, _, _) = ssh_error
    assert match(Command('ssh', errormsg))
    assert match(Command('ssh', errormsg))
    assert match(Command('scp something something', errormsg))
    assert match(Command('scp something something', errormsg))
    assert not match(Command(errormsg, ''))
    assert not match(Command('notssh', errormsg))
    assert not match(Command('ssh', ''))

@pytest.mark.skipif(os.name == 'nt', reason='Skip if testing on Windows')
def test_side_effect(ssh_error):
    if False:
        for i in range(10):
            print('nop')
    (errormsg, path, reset, known_hosts) = ssh_error
    command = Command('ssh user@host', errormsg)
    side_effect(command, None)
    expected = ['123.234.567.890 asdjkasjdakjsd\n', '111.222.333.444 qwepoiwqepoiss\n']
    assert known_hosts(path) == expected

def test_get_new_command(ssh_error, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    (errormsg, _, _, _) = ssh_error
    assert get_new_command(Command('ssh user@host', errormsg)) == 'ssh user@host'