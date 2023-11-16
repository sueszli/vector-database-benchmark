"""
Tests for file.touch function
"""
import pytest
from salt.exceptions import SaltInvocationError
pytestmark = [pytest.mark.windows_whitelisted]

@pytest.fixture(scope='module')
def file(modules):
    if False:
        return 10
    return modules.file

def test_touch(file, tmp_path):
    if False:
        return 10
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target))
    assert target.exists()

def test_touch_error_atime(file, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test touch with non int input\n    '
    target = tmp_path / 'test.file'
    with pytest.raises(SaltInvocationError) as exc:
        file.touch(str(target), atime='string')
    assert 'atime and mtime must be integers' in exc.value.message

def test_touch_error_mtime(file, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test touch with non int input\n    '
    target = tmp_path / 'test.file'
    with pytest.raises(SaltInvocationError) as exc:
        file.touch(str(target), mtime='string')
    assert 'atime and mtime must be integers' in exc.value.message

def test_touch_atime(file, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target), atime=123)
    assert target.stat().st_atime == 123

def test_touch_atime_zero(file, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target), atime=0)
    assert target.stat().st_atime == 0

def test_touch_mtime(file, tmp_path):
    if False:
        return 10
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target), mtime=234)
    assert target.stat().st_mtime == 234

def test_touch_mtime_zero(file, tmp_path):
    if False:
        print('Hello World!')
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target), mtime=0)
    assert target.stat().st_mtime == 0

def test_touch_atime_mtime(file, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Test touch with defaults\n    '
    target = tmp_path / 'test.file'
    file.touch(str(target), atime=456, mtime=789)
    assert target.stat().st_atime == 456
    assert target.stat().st_mtime == 789