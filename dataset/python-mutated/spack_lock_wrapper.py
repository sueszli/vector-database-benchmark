"""Tests for Spack's wrapper module around llnl.util.lock."""
import os
import pytest
from llnl.util.filesystem import getuid, group_ids
import spack.config
import spack.util.lock as lk

def test_disable_locking(tmpdir):
    if False:
        i = 10
        return i + 15
    'Ensure that locks do no real locking when disabled.'
    lock_path = str(tmpdir.join('lockfile'))
    lock = lk.Lock(lock_path, enable=False)
    lock.acquire_read()
    assert not os.path.exists(lock_path)
    lock.acquire_write()
    assert not os.path.exists(lock_path)
    lock.release_write()
    assert not os.path.exists(lock_path)
    lock.release_read()
    assert not os.path.exists(lock_path)

@pytest.mark.nomockstage
def test_lock_checks_user(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Ensure lock checks work with a self-owned, self-group repo.'
    uid = getuid()
    if uid not in group_ids():
        pytest.skip('user has no group with gid == uid')
    tmpdir.chown(uid, uid)
    path = str(tmpdir)
    tmpdir.chmod(484)
    lk.check_lock_safety(path)
    tmpdir.chmod(508)
    lk.check_lock_safety(path)
    tmpdir.chmod(511)
    with pytest.raises(spack.error.SpackError):
        lk.check_lock_safety(path)
    tmpdir.chmod(316)
    lk.check_lock_safety(path)
    tmpdir.chmod(319)
    lk.check_lock_safety(path)

@pytest.mark.nomockstage
def test_lock_checks_group(tmpdir):
    if False:
        return 10
    'Ensure lock checks work with a self-owned, non-self-group repo.'
    uid = getuid()
    gid = next((g for g in group_ids() if g != uid), None)
    if not gid:
        pytest.skip('user has no group with gid != uid')
    tmpdir.chown(uid, gid)
    path = str(tmpdir)
    tmpdir.chmod(484)
    lk.check_lock_safety(path)
    tmpdir.chmod(508)
    with pytest.raises(spack.error.SpackError):
        lk.check_lock_safety(path)
    tmpdir.chmod(511)
    with pytest.raises(spack.error.SpackError):
        lk.check_lock_safety(path)
    tmpdir.chmod(316)
    lk.check_lock_safety(path)
    tmpdir.chmod(319)
    lk.check_lock_safety(path)