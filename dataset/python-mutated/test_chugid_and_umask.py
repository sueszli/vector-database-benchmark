import functools
import os
import pathlib
import subprocess
import tempfile
import pytest
import salt.utils.user

def _check_skip(grains):
    if False:
        i = 10
        return i + 15
    if grains['os'] == 'MacOS':
        return True
    return False
pytestmark = [pytest.mark.destructive_test, pytest.mark.skip_if_not_root, pytest.mark.skip_on_windows, pytest.mark.skip_initial_gh_actions_failure(skip=_check_skip)]

@pytest.fixture(scope='module')
def account_1():
    if False:
        print('Hello World!')
    with pytest.helpers.create_account(create_group=True) as _account:
        yield _account

@pytest.fixture(scope='module')
def account_2(account_1):
    if False:
        while True:
            i = 10
    with pytest.helpers.create_account(group_name=account_1.group.name) as _account:
        yield _account

def test_chugid(account_1):
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = pathlib.Path(tmp_path)
        tmp_path.chmod(504)
        testfile = tmp_path / 'testfile'
        ret = subprocess.run(['touch', str(testfile)], preexec_fn=functools.partial(salt.utils.user.chugid_and_umask, runas=account_1.username, umask=None, group=None), check=False)
        assert ret.returncode != 0
        os.chown(str(tmp_path), 0, account_1.group.info.gid)
        ret = subprocess.run(['touch', str(testfile)], preexec_fn=functools.partial(salt.utils.user.chugid_and_umask, runas=account_1.username, umask=None, group=None), check=False)
        assert ret.returncode == 0
        assert testfile.exists()
        testfile_stat = testfile.stat()
        assert testfile_stat.st_uid == account_1.info.uid
        assert testfile_stat.st_gid == account_1.info.gid

def test_chugid_and_group(account_1, account_2, tmp_path):
    if False:
        return 10
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = pathlib.Path(tmp_path)
        tmp_path.chmod(504)
        testfile = tmp_path / 'testfile'
        ret = subprocess.run(['touch', str(testfile)], preexec_fn=functools.partial(salt.utils.user.chugid_and_umask, runas=account_2.username, umask=None, group=account_1.group.name), check=False)
        assert ret.returncode != 0
        os.chown(str(tmp_path), 0, account_1.group.info.gid)
        ret = subprocess.run(['touch', str(testfile)], preexec_fn=functools.partial(salt.utils.user.chugid_and_umask, runas=account_2.username, umask=None, group=account_1.group.name), check=False)
        assert ret.returncode == 0
        assert testfile.exists()
        testfile_stat = testfile.stat()
        assert testfile_stat.st_uid == account_2.info.uid
        assert testfile_stat.st_gid == account_1.group.info.gid