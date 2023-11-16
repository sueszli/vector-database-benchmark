import pathlib
import pytest
import salt.spm.pkgfiles.local as spm
import salt.syspaths
from tests.support.mock import MagicMock

@pytest.fixture()
def configure_loader_modules():
    if False:
        return 10
    return {spm: {'__opts__': {'spm_node_type': 'master'}}}

class MockTar:

    def __init__(self):
        if False:
            print('Hello World!')
        self.name = str(pathlib.Path('apache', '_README'))
        self.path = str(pathlib.Path(salt.syspaths.CACHE_DIR, 'master', 'extmods'))

def test_install_file(tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    test spm.pkgfiles.local\n    '
    assert spm.install_file('apache', formula_tar=MagicMock(), member=MockTar(), formula_def={'name': 'apache'}, conn={'formula_path': str(tmp_path / 'test')}) == MockTar().path