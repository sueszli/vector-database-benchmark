"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.augeas_cfg
"""
import pytest
import salt.modules.augeas_cfg as augeas_cfg
from salt.exceptions import SaltInvocationError
from tests.support.mock import MagicMock, patch
if augeas_cfg.HAS_AUGEAS:
    from augeas import Augeas as _Augeas
pytestmark = [pytest.mark.skipif(augeas_cfg.HAS_AUGEAS is False, reason='python-augeas is required for this test case')]

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {augeas_cfg: {}}

def test_execute():
    if False:
        while True:
            i = 10
    '\n    Test if it execute Augeas commands\n    '
    assert augeas_cfg.execute() == {'retval': True}

def test_execute_io_error():
    if False:
        print('Hello World!')
    '\n    Test if it execute Augeas commands\n    '
    ret = {'error': 'Command  is not supported (yet)', 'retval': False}
    assert augeas_cfg.execute(None, None, [' ']) == ret

def test_execute_value_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it execute Augeas commands\n    '
    ret = {'retval': False, 'error': 'Invalid formatted command, see debug log for details: '}
    assert augeas_cfg.execute(None, None, ['set ']) == ret

def test_get():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it get a value for a specific augeas path\n    '
    mock = MagicMock(side_effect=RuntimeError('error'))
    with patch.object(_Augeas, 'match', mock):
        assert augeas_cfg.get('/etc/hosts') == {'error': 'error'}
    mock = MagicMock(return_value=True)
    with patch.object(_Augeas, 'match', mock):
        assert augeas_cfg.get('/etc/hosts') == {'/etc/hosts': None}

def test_setvalue():
    if False:
        i = 10
        return i + 15
    '\n    Test if it set a value for a specific augeas path\n    '
    assert augeas_cfg.setvalue('prefix=/etc/hosts') == {'retval': True}

def test_setvalue_io_error():
    if False:
        while True:
            i = 10
    '\n    Test if it set a value for a specific augeas path\n    '
    mock = MagicMock(side_effect=IOError(''))
    with patch.object(_Augeas, 'save', mock):
        assert augeas_cfg.setvalue('prefix=/files/etc/') == {'retval': False, 'error': ''}

def test_setvalue_uneven_path():
    if False:
        print('Hello World!')
    '\n    Test if it set a value for a specific augeas path\n    '
    mock = MagicMock(side_effect=RuntimeError('error'))
    with patch.object(_Augeas, 'match', mock):
        pytest.raises(SaltInvocationError, augeas_cfg.setvalue, ['/files/etc/hosts/1/canonical', 'localhost'])

def test_setvalue_one_prefix():
    if False:
        print('Hello World!')
    '\n    Test if it set a value for a specific augeas path\n    '
    pytest.raises(SaltInvocationError, augeas_cfg.setvalue, 'prefix=/files', '10.18.1.1', 'prefix=/etc', 'test')

def test_match():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it matches for path expression\n    '
    assert augeas_cfg.match('/etc/service', 'ssh') == {}

def test_match_runtime_error():
    if False:
        while True:
            i = 10
    '\n    Test if it matches for path expression\n    '
    mock = MagicMock(side_effect=RuntimeError('error'))
    with patch.object(_Augeas, 'match', mock):
        assert augeas_cfg.match('/etc/service-name', 'ssh') == {}

def test_remove():
    if False:
        print('Hello World!')
    '\n    Test if it removes for path expression\n    '
    assert augeas_cfg.remove('/etc/service') == {'count': 0, 'retval': True}

def test_remove_io_runtime_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it removes for path expression\n    '
    mock = MagicMock(side_effect=RuntimeError('error'))
    with patch.object(_Augeas, 'save', mock):
        assert augeas_cfg.remove('/etc/service-name') == {'count': 0, 'error': 'error', 'retval': False}

def test_ls():
    if False:
        i = 10
        return i + 15
    '\n    Test if it list the direct children of a node\n    '
    assert augeas_cfg.ls('/etc/passwd') == {}

def test_tree():
    if False:
        return 10
    '\n    Test if it returns recursively the complete tree of a node\n    '
    assert augeas_cfg.tree('/etc/') == {'/etc': None}