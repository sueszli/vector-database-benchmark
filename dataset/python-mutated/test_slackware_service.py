"""
    :codeauthor: Piter Punk <piterpunk@slackware.com>
"""
import os
import pytest
import salt.modules.slackware_service as slackware_service
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {slackware_service: {}}

@pytest.fixture
def mocked_rcd():
    if False:
        print('Hello World!')
    glob_output = ['/etc/rc.d/rc.S', '/etc/rc.d/rc.M', '/etc/rc.d/rc.lxc', '/etc/rc.d/rc.modules', '/etc/rc.d/rc.ntpd', '/etc/rc.d/rc.rpc', '/etc/rc.d/rc.salt-master', '/etc/rc.d/rc.salt-minion', '/etc/rc.d/rc.something.conf', '/etc/rc.d/rc.sshd']
    access_output = [True, True, False, True, False, False]
    glob_mock = patch('glob.glob', autospec=True, return_value=glob_output)
    os_path_exists_mock = patch('os.path.exists', autospec=True, return_value=True)
    os_access_mock = patch('os.access', autospec=True, side_effect=access_output)
    return (glob_mock, os_path_exists_mock, os_access_mock)

def test_get_all_rc_services_minus_system_and_config_files(mocked_rcd):
    if False:
        i = 10
        return i + 15
    '\n    In Slackware, the services are started, stopped, enabled or disabled\n    using rc.service scripts under the /etc/rc.d directory.\n\n    This tests if only service rc scripts are returned by get_alli function.\n    System rc scripts (like rc.M) and configuration rc files (like\n    rc.something.conf) needs to be removed from output. Also, we remove the\n    leading "/etc/rc.d/rc." to output only the service names.\n\n    Return list: lxc ntpd rpc salt-master salt-minion sshd\n    '
    services_all = ['lxc', 'ntpd', 'rpc', 'salt-master', 'salt-minion', 'sshd']
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert slackware_service.get_all() == services_all

def test_if_only_executable_rc_files_are_returned_by_get_enabled(mocked_rcd):
    if False:
        return 10
    '\n    In Slackware, the services are enabled at boot by setting the executable\n    bit in their respective rc files.\n\n    This tests if all system rc scripts, configuration rc files and service rc\n    scripts without the executable bit set were filtered out from output.\n\n    Return list: lxc ntpd salt-master\n    '
    services_enabled = ['lxc', 'ntpd', 'salt-master']
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert slackware_service.get_enabled() == services_enabled

def test_if_only_not_executable_rc_files_are_returned_by_get_disabled(mocked_rcd):
    if False:
        for i in range(10):
            print('nop')
    '\n    In Slackware, the services are disabled at boot by unsetting the executable\n    bit in their respective rc files.\n\n    This tests if all system rc scripts, configuration rc files and service rc\n    scripts with the executable bit set were filtered out from output.\n\n    Return list: rpc salt-minion sshd\n    '
    services_disabled = ['rpc', 'salt-minion', 'sshd']
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert slackware_service.get_disabled() == services_disabled

def test_if_a_rc_service_file_in_rcd_is_listed_as_available(mocked_rcd):
    if False:
        print('Hello World!')
    '\n    Test if an existent service rc file with the rc.service name format is\n    present in rc.d directory and returned by "available" function\n    '
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert slackware_service.available('lxc')

def test_if_a_rc_service_file_not_in_rcd_is_not_listed_as_available(mocked_rcd):
    if False:
        print('Hello World!')
    '\n    Test if a non existent service rc file with the rc.service name format is\n    not present in rc.d directory and not returned by "available" function\n    '
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert not slackware_service.available('docker')

def test_if_a_rc_service_file_not_in_rcd_is_listed_as_missing(mocked_rcd):
    if False:
        return 10
    '\n    Test if a non existent service rc file with the rc.service name format is\n    not present in rc.d directory and returned by "missing" function\n    '
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert slackware_service.missing('docker')

def test_if_a_rc_service_file_in_rcd_is_not_listed_as_missing(mocked_rcd):
    if False:
        i = 10
        return i + 15
    '\n    Test if an existent service rc file with the rc.service name format is\n    present in rc.d directory and not returned by "missing" function\n    '
    (glob_mock, os_path_exists_mock, os_access_mock) = mocked_rcd
    with glob_mock, os_path_exists_mock, os_access_mock:
        assert not slackware_service.missing('lxc')

def test_service_start():
    if False:
        print('Hello World!')
    '\n    Test for Start the specified service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.start('name')

def test_service_stop():
    if False:
        i = 10
        return i + 15
    '\n    Test for Stop the specified service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.stop('name')

def test_service_restart():
    if False:
        print('Hello World!')
    '\n    Test for Restart the named service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.restart('name')

def test_service_reload_():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Reload the named service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.reload_('name')

def test_service_force_reload():
    if False:
        while True:
            i = 10
    '\n    Test for Force-reload the named service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.force_reload('name')

def test_service_status():
    if False:
        print('Hello World!')
    '\n    Test for Return the status for a service\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(slackware_service.__salt__, {'cmd.retcode': mock}):
        assert not slackware_service.status('name')

def test_if_executable_bit_is_set_when_enable_a_disabled_service():
    if False:
        for i in range(10):
            print('nop')
    '\n    In Slackware, the services are enabled at boot by setting the executable\n    bit in their respective rc files.\n\n    This tests if, given a disabled rc file with permissions 0644, we enable it by\n    changing its permissions to 0755\n    '
    os_path_exists_mock = patch('os.path.exists', autospec=True, return_value=True)
    os_chmod = MagicMock(autospec=True, return_value=True)
    os_chmod_mock = patch('os.chmod', os_chmod)
    os_stat_result = os.stat_result((33188, 142555, 64770, 1, 0, 0, 1340, 1597376187, 1597376188, 1597376189))
    os_stat_mock = patch('os.stat', autospec=True, return_value=os_stat_result)
    with os_path_exists_mock, os_chmod_mock, os_stat_mock:
        slackware_service.enable('svc_to_enable')
        os_chmod.assert_called_with('/etc/rc.d/rc.svc_to_enable', 33261)

def test_if_executable_bit_is_unset_when_disable_an_enabled_service():
    if False:
        return 10
    '\n    In Slackware, the services are disabled at boot by unsetting the executable\n    bit in their respective rc files.\n\n    This tests if, given an enabled rc file with permissions 0755, we disable it by\n    changing its permissions to 0644\n    '
    os_path_exists_mock = patch('os.path.exists', autospec=True, return_value=True)
    os_chmod = MagicMock(autospec=True, return_value=True)
    os_chmod_mock = patch('os.chmod', os_chmod)
    os_stat_result = os.stat_result((33261, 142555, 64770, 1, 0, 0, 1340, 1597376187, 1597376188, 1597376189))
    os_stat_mock = patch('os.stat', autospec=True, return_value=os_stat_result)
    with os_path_exists_mock, os_chmod_mock, os_stat_mock:
        slackware_service.disable('svc_to_disable')
        os_chmod.assert_called_with('/etc/rc.d/rc.svc_to_disable', 33188)

def test_if_an_enabled_service_is_not_disabled():
    if False:
        i = 10
        return i + 15
    "\n    A service can't be enabled and disabled at same time.\n\n    This tests if a service that returns True to enabled returns False to disabled\n    "
    os_path_exists_mock = patch('os.path.exists', autospec=True, return_value=True)
    os_access_mock = patch('os.access', autospec=True, return_value=True)
    with os_path_exists_mock, os_access_mock:
        assert slackware_service.enabled('lxc')
        assert not slackware_service.disabled('lxc')

def test_if_a_disabled_service_is_not_enabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    A service can't be enabled and disabled at same time.\n\n    This tests if a service that returns True to disabled returns False to enabled\n    "
    os_path_exists_mock = patch('os.path.exists', autospec=True, return_value=True)
    os_access_mock = patch('os.access', autospec=True, return_value=False)
    with os_path_exists_mock, os_access_mock:
        assert slackware_service.disabled('rpc')
        assert not slackware_service.enabled('rpc')