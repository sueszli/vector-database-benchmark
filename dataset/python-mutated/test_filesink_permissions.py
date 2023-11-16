import os
from stat import S_IMODE
import pytest
from loguru import logger

@pytest.fixture(scope='module', autouse=True)
def set_umask():
    if False:
        while True:
            i = 10
    default = os.umask(0)
    yield
    os.umask(default)

@pytest.mark.parametrize('permissions', [511, 502, 484, 448, 393])
def test_log_file_permissions(tmp_path, permissions):
    if False:
        while True:
            i = 10

    def file_permission_opener(file, flags):
        if False:
            while True:
                i = 10
        return os.open(file, flags, permissions)
    filepath = tmp_path / 'file.log'
    logger.add(filepath, opener=file_permission_opener)
    logger.debug('Message')
    stat_result = os.stat(str(filepath))
    expected = 438 if os.name == 'nt' else permissions
    assert S_IMODE(stat_result.st_mode) == expected

@pytest.mark.parametrize('permissions', [511, 502, 484, 448, 393])
def test_rotation_permissions(tmp_path, permissions, set_umask):
    if False:
        i = 10
        return i + 15

    def file_permission_opener(file, flags):
        if False:
            while True:
                i = 10
        return os.open(file, flags, permissions)
    logger.add(tmp_path / 'file.log', rotation=0, opener=file_permission_opener)
    logger.debug('Message')
    files = list(tmp_path.iterdir())
    assert len(files) == 2
    for filepath in files:
        stat_result = os.stat(str(filepath))
        expected = 438 if os.name == 'nt' else permissions
        assert S_IMODE(stat_result.st_mode) == expected