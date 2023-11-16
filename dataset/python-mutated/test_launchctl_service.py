"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.launchctl
"""
import pytest
import salt.modules.launchctl_service as launchctl
import salt.utils.stringutils
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {launchctl: {}}

def test_get_all():
    if False:
        while True:
            i = 10
    '\n    Test for Return all installed services\n    '
    with patch.dict(launchctl.__salt__, {'cmd.run': MagicMock(return_value='A\tB\tC\t\n')}):
        with patch.object(launchctl, '_available_services', return_value={'A': 'a', 'B': 'b'}):
            assert launchctl.get_all() == ['A', 'B', 'C']

def test_available():
    if False:
        print('Hello World!')
    '\n    Test for Check that the given service is available.\n    '
    with patch.object(launchctl, '_service_by_name', return_value=True):
        assert launchctl.available('job_label')

def test_missing():
    if False:
        i = 10
        return i + 15
    '\n    Test for The inverse of service.available\n    '
    with patch.object(launchctl, '_service_by_name', return_value=True):
        assert not launchctl.missing('job_label')

def test_status():
    if False:
        print('Hello World!')
    '\n    Test for Return the status for a service\n    '
    launchctl_data = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0">\n<dict>\n    <key>Label</key>\n    <string>salt-minion</string>\n    <key>LastExitStatus</key>\n    <integer>0</integer>\n    <key>LimitLoadToSessionType</key>\n    <string>System</string>\n    <key>OnDemand</key>\n    <false/>\n    <key>PID</key>\n    <integer>71</integer>\n    <key>ProgramArguments</key>\n    <array>\n        <string>/usr/local/bin/salt-minion</string>\n    </array>\n    <key>TimeOut</key>\n    <integer>30</integer>\n</dict>\n</plist>'
    with patch.object(launchctl, '_service_by_name', return_value={'plist': {'Label': 'A'}}):
        launchctl_data = salt.utils.stringutils.to_bytes(launchctl_data)
        with patch.object(launchctl, '_get_launchctl_data', return_value=launchctl_data):
            assert launchctl.status('job_label')

def test_stop():
    if False:
        while True:
            i = 10
    '\n    Test for Stop the specified service\n    '
    with patch.object(launchctl, '_service_by_name', return_value={'file_path': 'A'}):
        with patch.dict(launchctl.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
            assert launchctl.stop('job_label')
    with patch.object(launchctl, '_service_by_name', return_value=None):
        assert not launchctl.stop('job_label')

def test_start():
    if False:
        print('Hello World!')
    '\n    Test for Start the specified service\n    '
    with patch.object(launchctl, '_service_by_name', return_value={'file_path': 'A'}):
        with patch.dict(launchctl.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
            assert launchctl.start('job_label')
    with patch.object(launchctl, '_service_by_name', return_value=None):
        assert not launchctl.start('job_label')

def test_restart():
    if False:
        return 10
    '\n    Test for Restart the named service\n    '
    with patch.object(launchctl, 'stop', return_value=None):
        with patch.object(launchctl, 'start', return_value=True):
            assert launchctl.restart('job_label')