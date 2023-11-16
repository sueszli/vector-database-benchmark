"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.rdp
"""
import pytest
import salt.modules.rdp as rdp
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {rdp: {}}

def test_enable():
    if False:
        i = 10
        return i + 15
    '\n    Test if it enables RDP the service on the server\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(rdp.__salt__, {'cmd.run': mock}), patch('salt.modules.rdp._parse_return_code_powershell', MagicMock(return_value=0)):
        assert rdp.enable()

def test_disable():
    if False:
        i = 10
        return i + 15
    '\n    Test if it disables RDP the service on the server\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(rdp.__salt__, {'cmd.run': mock}), patch('salt.modules.rdp._parse_return_code_powershell', MagicMock(return_value=0)):
        assert rdp.disable()

def test_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it shows rdp is enabled on the server\n    '
    mock = MagicMock(return_value='1')
    with patch.dict(rdp.__salt__, {'cmd.run': mock}):
        assert rdp.status()