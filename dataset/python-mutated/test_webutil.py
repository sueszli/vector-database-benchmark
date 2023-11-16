"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.webutil
"""
import pytest
import salt.modules.webutil as htpasswd
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {htpasswd: {}}

def test_useradd():
    if False:
        return 10
    '\n    Test if it adds an HTTP user using the htpasswd command\n    '
    mock = MagicMock(return_value={'out': 'Salt'})
    with patch.dict(htpasswd.__salt__, {'cmd.run_all': mock}), patch('os.path.exists', MagicMock(return_value=True)):
        assert htpasswd.useradd('/etc/httpd/htpasswd', 'larry', 'badpassword') == {'out': 'Salt'}

def test_userdel():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it delete an HTTP user from the specified htpasswd file.\n    '
    mock = MagicMock(return_value='Salt')
    with patch.dict(htpasswd.__salt__, {'cmd.run': mock}), patch('os.path.exists', MagicMock(return_value=True)):
        assert htpasswd.userdel('/etc/httpd/htpasswd', 'larry') == ['Salt']

def test_userdel_missing_htpasswd():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it returns error when no htpasswd file exists\n    '
    with patch('os.path.exists', MagicMock(return_value=False)):
        assert htpasswd.userdel('/etc/httpd/htpasswd', 'larry') == 'Error: The specified htpasswd file does not exist'