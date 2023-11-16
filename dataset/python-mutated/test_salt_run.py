import re
import pytest
import salt.defaults.exitcodes
import salt.utils.files
import salt.utils.platform
import salt.utils.pycrypto
import salt.utils.yaml
pytestmark = [pytest.mark.core_test, pytest.mark.windows_whitelisted]

@pytest.fixture
def salt_run_cli(salt_master):
    if False:
        while True:
            i = 10
    '\n    Override salt_run_cli fixture to provide an increased default_timeout to the calls\n    '
    return salt_master.salt_run_cli(timeout=120)

def test_in_docs(salt_run_cli):
    if False:
        for i in range(10):
            print('nop')
    '\n    test the salt-run docs system\n    '
    ret = salt_run_cli.run('-d')
    assert 'jobs.active:' in ret.stdout
    assert 'jobs.list_jobs:' in ret.stdout
    assert 'jobs.lookup_jid:' in ret.stdout
    assert 'manage.down:' in ret.stdout
    assert 'manage.up:' in ret.stdout
    assert 'network.wol:' in ret.stdout
    assert 'network.wollist:' in ret.stdout

def test_not_in_docs(salt_run_cli):
    if False:
        for i in range(10):
            print('nop')
    '\n    test the salt-run docs system\n    '
    ret = salt_run_cli.run('-d')
    assert 'jobs.SaltException:' not in ret.stdout

def test_salt_documentation_too_many_arguments(salt_run_cli):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to see if passing additional arguments shows an error\n    '
    ret = salt_run_cli.run('-d', 'virt.list', 'foo')
    assert ret.returncode != 0
    assert 'You can only get documentation for one method at one time' in ret.stderr

def test_exit_status_unknown_argument(salt_run_cli):
    if False:
        print('Hello World!')
    '\n    Ensure correct exit status when an unknown argument is passed to salt-run.\n    '
    ret = salt_run_cli.run('--unknown-argument')
    assert ret.returncode == salt.defaults.exitcodes.EX_USAGE, ret
    assert 'Usage' in ret.stderr
    assert 'no such option: --unknown-argument' in ret.stderr

def test_exit_status_correct_usage(salt_run_cli):
    if False:
        return 10
    '\n    Ensure correct exit status when salt-run starts correctly.\n    '
    ret = salt_run_cli.run('test.arg', 'arg1', kwarg1='kwarg1')
    assert ret.returncode == salt.defaults.exitcodes.EX_OK, ret

@pytest.mark.skip_if_not_root
@pytest.mark.parametrize('flag', ['--auth', '--eauth', '--external-auth', '-a'])
@pytest.mark.skip_on_windows(reason='PAM is not supported on Windows')
def test_salt_run_with_eauth_all_args(salt_run_cli, salt_eauth_account, flag):
    if False:
        i = 10
        return i + 15
    '\n    test salt-run with eauth\n    tests all eauth args\n    '
    ret = salt_run_cli.run(flag, 'pam', '--username', salt_eauth_account.username, '--password', salt_eauth_account.password, 'test.arg', 'arg', kwarg='kwarg1', _timeout=240)
    assert ret.returncode == 0, ret
    assert ret.data, ret
    expected = {'args': ['arg'], 'kwargs': {'kwarg': 'kwarg1'}}
    assert ret.data == expected, ret

@pytest.mark.skip_if_not_root
@pytest.mark.skip_on_windows(reason='PAM is not supported on Windows')
def test_salt_run_with_eauth_bad_passwd(salt_run_cli, salt_eauth_account):
    if False:
        i = 10
        return i + 15
    '\n    test salt-run with eauth and bad password\n    '
    ret = salt_run_cli.run('-a', 'pam', '--username', salt_eauth_account.username, '--password', 'wrongpassword', 'test.arg', 'arg', kwarg='kwarg1')
    assert ret.stdout == 'Authentication failure of type "eauth" occurred for user {}.'.format(salt_eauth_account.username)

@pytest.mark.skip_if_not_root
def test_salt_run_with_wrong_eauth(salt_run_cli, salt_eauth_account):
    if False:
        while True:
            i = 10
    '\n    test salt-run with wrong eauth parameter\n    '
    ret = salt_run_cli.run('-a', 'wrongeauth', '--username', salt_eauth_account.username, '--password', salt_eauth_account.password, 'test.arg', 'arg', kwarg='kwarg1')
    assert ret.returncode == 0, ret
    assert re.search('^The specified external authentication system \\"wrongeauth\\" is not available\\nAvailable eauth types: auto, .*', ret.stdout.replace('\r\n', '\n'))