import pytest
import salt.modules.cmdmod as cmdmod

@pytest.fixture(scope='module')
def account():
    if False:
        i = 10
        return i + 15
    with pytest.helpers.create_account(create_group=True) as _account:
        yield _account

@pytest.fixture(scope='module')
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {cmdmod: {'__grains__': {'os': 'linux', 'os_family': 'linux'}}}

@pytest.mark.skip_on_windows
@pytest.mark.skip_if_not_root
def test_run_as(account):
    if False:
        while True:
            i = 10
    ret = cmdmod.run('id', runas=account.username)
    assert 'gid={}'.format(account.info.gid) in ret
    assert 'uid={}'.format(account.info.uid) in ret