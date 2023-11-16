"""Test LockFile functionality."""
from __future__ import print_function
from collections import namedtuple
from multiprocessing import Pool
import os
import shutil
import sys
import tempfile
import traceback
import pytest
from workflow.util import AcquisitionError, LockFile
from workflow.workflow import Settings
Paths = namedtuple('Paths', 'testfile lockfile')

@pytest.fixture(scope='function')
def paths(request):
    if False:
        print('Hello World!')
    'Test and lock file paths.'
    tempdir = tempfile.mkdtemp()
    testfile = os.path.join(tempdir, 'myfile.txt')

    def rm():
        if False:
            return 10
        shutil.rmtree(tempdir)
    request.addfinalizer(rm)
    return Paths(testfile, testfile + '.lock')

def test_lockfile_created(paths):
    if False:
        while True:
            i = 10
    'Lock file created and deleted.'
    assert not os.path.exists(paths.testfile)
    assert not os.path.exists(paths.lockfile)
    with LockFile(paths.testfile, timeout=0.2) as lock:
        assert lock.locked
        assert os.path.exists(paths.lockfile)
    assert not os.path.exists(paths.lockfile)

def test_sequential_access(paths):
    if False:
        for i in range(10):
            print('nop')
    'Sequential access to locked file.'
    assert not os.path.exists(paths.testfile)
    assert not os.path.exists(paths.lockfile)
    lock = LockFile(paths.testfile, 0.1)
    with lock:
        assert lock.locked
        assert not lock.acquire(False)
        with pytest.raises(AcquisitionError):
            lock.acquire(True)
    assert lock.release() is False
    assert not os.path.exists(paths.lockfile)

def _write_test_data(args):
    if False:
        return 10
    'Write 10 lines to the test file.'
    (paths, data) = args
    for i in range(10):
        with LockFile(paths.testfile, 0.5) as lock:
            assert lock.locked
            with open(paths.testfile, 'a') as fp:
                fp.write(data + '\n')

def test_concurrent_access(paths):
    if False:
        i = 10
        return i + 15
    'Concurrent access to locked file is serialised.'
    assert not os.path.exists(paths.testfile)
    assert not os.path.exists(paths.lockfile)
    lock = LockFile(paths.testfile, 0.5)
    pool = Pool(5)
    pool.map(_write_test_data, [(paths, str(i) * 20) for i in range(1, 6)])
    assert not lock.locked
    assert not os.path.exists(paths.lockfile)
    with open(paths.testfile) as fp:
        lines = [line.strip() for line in fp.readlines()]
    for line in lines:
        assert len(set(line)) == 1

def _write_settings(args):
    if False:
        i = 10
        return i + 15
    'Write a new value to the Settings.'
    (paths, key, value) = args
    try:
        s = Settings(paths.testfile)
        s[key] = value
        print('Settings[{0}] = {1}'.format(key, value))
    except Exception as err:
        print('error opening settings (%s): %s' % (key, traceback.format_exc()), file=sys.stderr)
        return err

def test_concurrent_settings(paths):
    if False:
        i = 10
        return i + 15
    'Concurrent access to Settings is serialised.'
    assert not os.path.exists(paths.testfile)
    assert not os.path.exists(paths.lockfile)
    defaults = {'foo': 'bar'}
    Settings(paths.testfile, defaults)
    data = [(paths, 'thread_{0}'.format(i), 'value_{0}'.format(i)) for i in range(1, 10)]
    pool = Pool(5)
    errs = pool.map(_write_settings, data)
    errs = [e for e in errs if e is not None]
    assert errs == []
    s = Settings(paths.testfile)
    assert s['foo'] == 'bar'
    assert len(s) > 1
if __name__ == '__main__':
    pytest.main([__file__])