"""
    :codeauthor: :email:`Simon Dodsley <simon@purestorage.com>`
"""
import pytest
import salt.modules.purefb as purefb
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {purefb: {}}

def test_fs_create():
    if False:
        i = 10
        return i + 15
    '\n    Test for creation of a filesystem\n    '
    with patch.object(purefb, 'fs_create', return_value=True):
        assert purefb.fs_create('test') is True

def test_fs_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test for deletion of a filesystem\n    '
    with patch.object(purefb, 'fs_delete', return_value=True):
        assert purefb.fs_delete('test') is True

def test_fs_eradicate():
    if False:
        print('Hello World!')
    '\n    Test for eradication of a filesystem\n    '
    with patch.object(purefb, 'fs_eradicate', return_value=True):
        assert purefb.fs_eradicate('test') is True

def test_fs_extend():
    if False:
        return 10
    '\n    Test for size extention of a filesystem\n    '
    with patch.object(purefb, 'fs_extend', return_value=True):
        assert purefb.fs_extend('test', '33G') is True

def test_snap_create():
    if False:
        while True:
            i = 10
    '\n    Test for creation of a filesystem snapshot\n    '
    with patch.object(purefb, 'snap_create', return_value=True):
        assert purefb.snap_create('test', suffix='suffix') is True

def test_snap_delete():
    if False:
        while True:
            i = 10
    '\n    Test for deletion of a filesystem snapshot\n    '
    with patch.object(purefb, 'snap_delete', return_value=True):
        assert purefb.snap_delete('test', suffix='suffix') is True

def test_snap_eradicate():
    if False:
        return 10
    '\n    Test for eradication of a deleted filesystem snapshot\n    '
    with patch.object(purefb, 'snap_eradicate', return_value=True):
        assert purefb.snap_eradicate('test', suffix='suffix') is True