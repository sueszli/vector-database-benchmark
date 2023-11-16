"""
Test secure deletion utilities in securedrop/rm.py
"""
import os
import pytest
import rm

def test_secure_delete_capability(config):
    if False:
        i = 10
        return i + 15
    assert rm.check_secure_delete_capability() is True
    path = os.environ['PATH']
    try:
        os.environ['PATH'] = f'{config.TEMP_DIR}'
        assert rm.check_secure_delete_capability() is False
        fakeshred = os.path.join(config.TEMP_DIR, 'shred')
        with open(fakeshred, 'w') as f:
            f.write('#!/bin/bash\nexit 1\n')
        os.chmod(fakeshred, 448)
        assert rm.check_secure_delete_capability() is False
    finally:
        os.environ['PATH'] = path

def test_shred(config):
    if False:
        for i in range(10):
            print('nop')
    testfile = 'test_shred.txt'
    content = 'abc123\n'
    with pytest.raises(EnvironmentError):
        rm.shred(os.path.abspath(os.path.join(config.TEMP_DIR, 'nonexistentshredtarget')))
    d = os.path.abspath(os.path.join(config.TEMP_DIR, 'nonexistentshredtarget'))
    os.makedirs(d)
    with pytest.raises(ValueError):
        rm.shred(d)
    os.rmdir(d)
    with open(testfile, 'w') as f:
        f.write(content)
    with open(testfile) as f:
        read_content = f.read()
        assert read_content == content
    rm.shred(testfile, delete=False)
    with open(testfile) as f:
        read_content = f.read()
        assert read_content != content
    rm.shred(testfile)
    assert os.path.exists(testfile) is False

def test_secure_delete(config):
    if False:
        return 10
    content = 'abc123\n'
    testfile = 'test_shred.txt'
    testfile1 = os.path.abspath(os.path.join(config.TEMP_DIR, testfile))
    with open(testfile1, 'w') as f:
        f.write(content)
    assert os.path.exists(testfile1)
    rm.secure_delete(testfile1)
    assert os.path.exists(testfile1) is False
    testdir = os.path.abspath(os.path.join(config.TEMP_DIR, 'shredtest1'))
    testsubdir1 = os.path.abspath(os.path.join(testdir, 'shredtest1.1'))
    testsubdir2 = os.path.abspath(os.path.join(testdir, 'shredtest1.2'))
    os.makedirs(testsubdir1)
    os.makedirs(testsubdir2)
    testfile1 = os.path.abspath(os.path.join(testdir, testfile))
    with open(testfile1, 'w') as f:
        f.write(content)
    testfile2 = os.path.abspath(os.path.join(testsubdir1, testfile))
    with open(testfile2, 'w') as f:
        f.write(content)
    assert os.path.exists(testfile1)
    assert os.path.exists(testfile2)
    rm.secure_delete(testdir)
    assert os.path.exists(testfile1) is False
    assert os.path.exists(testfile2) is False
    assert os.path.exists(testsubdir1) is False
    assert os.path.exists(testsubdir2) is False
    assert os.path.exists(testdir) is False