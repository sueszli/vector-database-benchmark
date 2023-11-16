import pytest
import salt.fileserver.svnfs as svnfs
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {svnfs: {}}

def test_env_is_exposed():
    if False:
        print('Hello World!')
    '\n    test _env_is_exposed method when\n    base is in whitelist\n    '
    with patch.dict(svnfs.__opts__, {'svnfs_saltenv_whitelist': 'base', 'svnfs_saltenv_blacklist': ''}):
        assert svnfs._env_is_exposed('base')

def test_env_is_exposed_blacklist():
    if False:
        while True:
            i = 10
    '\n    test _env_is_exposed method when\n    base is in blacklist\n    '
    with patch.dict(svnfs.__opts__, {'svnfs_saltenv_whitelist': '', 'svnfs_saltenv_blacklist': 'base'}):
        assert not svnfs._env_is_exposed('base')