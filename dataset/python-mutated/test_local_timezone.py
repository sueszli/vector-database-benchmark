from __future__ import annotations
import os
import sys
import pytest
from pendulum.tz.local_timezone import _get_unix_timezone
from pendulum.tz.local_timezone import _get_windows_timezone

@pytest.mark.skipif(sys.platform == 'win32', reason='Test only available for UNIX systems')
def test_unix_symlink():
    if False:
        for i in range(10):
            print('nop')
    local_path = os.path.join(os.path.split(__file__)[0], '..')
    tz = _get_unix_timezone(_root=os.path.join(local_path, 'fixtures', 'tz', 'symlink'))
    assert tz.name == 'Europe/Paris'

@pytest.mark.skipif(sys.platform == 'win32', reason='Test only available for UNIX systems')
def test_unix_clock():
    if False:
        while True:
            i = 10
    local_path = os.path.join(os.path.split(__file__)[0], '..')
    tz = _get_unix_timezone(_root=os.path.join(local_path, 'fixtures', 'tz', 'clock'))
    assert tz.name == 'Europe/Zurich'

@pytest.mark.skipif(sys.platform != 'win32', reason='Test only available for Windows')
def test_windows_timezone():
    if False:
        return 10
    timezone = _get_windows_timezone()
    assert timezone is not None

@pytest.mark.skipif(sys.platform == 'win32', reason='Test only available for UNIX systems')
def test_unix_etc_timezone_dir():
    if False:
        return 10
    local_path = os.path.join(os.path.split(__file__)[0], '..')
    root_path = os.path.join(local_path, 'fixtures', 'tz', 'timezone_dir')
    tz = _get_unix_timezone(_root=root_path)
    assert tz.name == 'Europe/Paris'