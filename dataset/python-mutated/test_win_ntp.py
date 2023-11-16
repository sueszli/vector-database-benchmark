"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.win_ntp as win_ntp
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {win_ntp: {}}

def test_set_servers():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it set Windows to use a list of NTP servers\n    '
    mock_service = MagicMock(return_value=False)
    with patch.dict(win_ntp.__salt__, {'service.status': mock_service, 'service.start': mock_service}):
        assert not win_ntp.set_servers('pool.ntp.org')
    mock_service = MagicMock(return_value=True)
    mock_cmd = MagicMock(side_effect=['Failure', 'Failure', 'Failure', 'NtpServer: time.windows.com,0x8'])
    with patch.dict(win_ntp.__salt__, {'service.status': mock_service, 'cmd.run': mock_cmd}):
        assert not win_ntp.set_servers('pool.ntp.org')
    mock_cmd = MagicMock(side_effect=['Success', 'Success', 'Success', 'NtpServer: pool.ntp.org'])
    with patch.dict(win_ntp.__salt__, {'service.status': mock_service, 'service.restart': mock_service, 'cmd.run': mock_cmd}):
        assert win_ntp.set_servers('pool.ntp.org')

def test_get_servers():
    if False:
        while True:
            i = 10
    '\n    Test if it get list of configured NTP servers\n    '
    mock_cmd = MagicMock(side_effect=['', 'NtpServer: SALT', 'NtpServer'])
    with patch.dict(win_ntp.__salt__, {'cmd.run': mock_cmd}):
        assert not win_ntp.get_servers()
        assert win_ntp.get_servers() == ['SALT']
        assert not win_ntp.get_servers()