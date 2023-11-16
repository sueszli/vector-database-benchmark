"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.hg
"""
import pytest
import salt.modules.hg as hg
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {hg: {}}

def test_revision():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Returns the long hash of a given identifier\n    '
    mock = MagicMock(side_effect=[{'retcode': 0, 'stdout': 'A'}, {'retcode': 1, 'stdout': 'A'}])
    with patch.dict(hg.__salt__, {'cmd.run_all': mock}):
        assert hg.revision('cwd') == 'A'
        assert hg.revision('cwd') == ''

def test_describe():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Mimic git describe.\n    '
    with patch.dict(hg.__salt__, {'cmd.run_stdout': MagicMock(return_value='A')}):
        with patch.object(hg, 'revision', return_value=False):
            assert hg.describe('cwd') == 'A'

def test_archive():
    if False:
        i = 10
        return i + 15
    '\n    Test for Export a tarball from the repository\n    '
    with patch.dict(hg.__salt__, {'cmd.run': MagicMock(return_value='A')}):
        assert hg.archive('cwd', 'output') == 'A'

def test_pull():
    if False:
        i = 10
        return i + 15
    '\n    Test for Perform a pull on the given repository\n    '
    with patch.dict(hg.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 0, 'stdout': 'A'})}):
        assert hg.pull('cwd') == 'A'

def test_update():
    if False:
        print('Hello World!')
    '\n    Test for Update to a given revision\n    '
    with patch.dict(hg.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 0, 'stdout': 'A'})}):
        assert hg.update('cwd', 'rev') == 'A'

def test_clone():
    if False:
        i = 10
        return i + 15
    '\n    Test for Clone a new repository\n    '
    with patch.dict(hg.__salt__, {'cmd.run_all': MagicMock(return_value={'retcode': 0, 'stdout': 'A'})}):
        assert hg.clone('cwd', 'repository') == 'A'

def test_status_single():
    if False:
        print('Hello World!')
    '\n    Test for Status to a given repository\n    '
    with patch.dict(hg.__salt__, {'cmd.run_stdout': MagicMock(return_value='A added 0\nA added 1\nM modified')}):
        assert hg.status('cwd') == {'added': ['added 0', 'added 1'], 'modified': ['modified']}

def test_status_multiple():
    if False:
        print('Hello World!')
    '\n    Test for Status to a given repository (cwd is list)\n    '
    with patch.dict(hg.__salt__, {'cmd.run_stdout': MagicMock(side_effect=lambda *args, **kwargs: {'dir 0': 'A file 0\n', 'dir 1': 'M file 1'}[kwargs['cwd']])}):
        assert hg.status(['dir 0', 'dir 1']) == {'dir 0': {'added': ['file 0']}, 'dir 1': {'modified': ['file 1']}}