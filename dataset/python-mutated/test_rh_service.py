"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.rh_service
"""
import textwrap
import pytest
import salt.modules.rh_service as rh_service
from tests.support.mock import MagicMock, patch

@pytest.fixture
def RET():
    if False:
        for i in range(10):
            print('nop')
    return ['hostname', 'mountall', 'network-interface', 'network-manager', 'salt-api', 'salt-master', 'salt-minion']

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {rh_service: {'_upstart_disable': None, '_upstart_enable': None, '_upstart_is_enabled': None}}

def _m_lst():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return value for [].\n    '
    return MagicMock(return_value=[])

def _m_ret(RET):
    if False:
        return 10
    '\n    Return value for RET.\n    '
    return MagicMock(return_value=RET)

def _m_bool(bol=True):
    if False:
        return 10
    '\n    Return Bool value.\n    '
    return MagicMock(return_value=bol)

def test__chkconfig_is_enabled():
    if False:
        print('Hello World!')
    '\n    test _chkconfig_is_enabled function\n    '
    name = 'atd'
    chkconfig_out = textwrap.dedent('\n        {}           0:off   1:off   2:off   3:on    4:on    5:on    6:off\n        '.format(name))
    xinetd_out = textwrap.dedent('        xinetd based services:\n                {}  on\n        '.format(name))
    with patch.object(rh_service, '_runlevel', MagicMock(return_value=3)):
        mock_run = MagicMock(return_value={'retcode': 0, 'stdout': chkconfig_out})
        with patch.dict(rh_service.__salt__, {'cmd.run_all': mock_run}):
            assert rh_service._chkconfig_is_enabled(name)
            assert not rh_service._chkconfig_is_enabled(name, 2)
            assert rh_service._chkconfig_is_enabled(name, 3)
        mock_run = MagicMock(return_value={'retcode': 0, 'stdout': xinetd_out})
        with patch.dict(rh_service.__salt__, {'cmd.run_all': mock_run}):
            assert rh_service._chkconfig_is_enabled(name)
            assert rh_service._chkconfig_is_enabled(name, 2)
            assert rh_service._chkconfig_is_enabled(name, 3)

def test_get_enabled(RET):
    if False:
        print('Hello World!')
    '\n    Test if it return the enabled services. Use the ``limit``\n    param to restrict results to services of that type.\n    '
    with patch.object(rh_service, '_upstart_services', _m_ret(RET)):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=False)):
            assert rh_service.get_enabled('upstart') == []
    mock_run = MagicMock(return_value='salt stack')
    with patch.dict(rh_service.__salt__, {'cmd.run': mock_run}):
        with patch.object(rh_service, '_sysv_services', _m_ret(RET)):
            with patch.object(rh_service, '_sysv_is_enabled', _m_bool()):
                assert rh_service.get_enabled('sysvinit') == RET
                with patch.object(rh_service, '_upstart_services', _m_lst()):
                    with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=True)):
                        assert rh_service.get_enabled() == RET

def test_get_disabled(RET):
    if False:
        while True:
            i = 10
    '\n    Test if it return the disabled services. Use the ``limit``\n    param to restrict results to services of that type.\n    '
    with patch.object(rh_service, '_upstart_services', _m_ret(RET)):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=False)):
            assert rh_service.get_disabled('upstart') == RET
    mock_run = MagicMock(return_value='salt stack')
    with patch.dict(rh_service.__salt__, {'cmd.run': mock_run}):
        with patch.object(rh_service, '_sysv_services', _m_ret(RET)):
            with patch.object(rh_service, '_sysv_is_enabled', _m_bool(False)):
                assert rh_service.get_disabled('sysvinit') == RET
                with patch.object(rh_service, '_upstart_services', _m_lst()):
                    with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=False)):
                        assert rh_service.get_disabled() == RET

def test_get_all(RET):
    if False:
        i = 10
        return i + 15
    '\n    Test if it return all installed services. Use the ``limit``\n    param to restrict results to services of that type.\n    '
    with patch.object(rh_service, '_upstart_services', _m_ret(RET)):
        assert rh_service.get_all('upstart') == RET
    with patch.object(rh_service, '_sysv_services', _m_ret(RET)):
        assert rh_service.get_all('sysvinit') == RET
        with patch.object(rh_service, '_upstart_services', _m_lst()):
            assert rh_service.get_all() == RET

def test_available():
    if False:
        while True:
            i = 10
    '\n    Test if it return True if the named service is available.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        assert rh_service.available('salt-api', 'upstart')
    with patch.object(rh_service, '_service_is_sysv', _m_bool()):
        assert rh_service.available('salt-api', 'sysvinit')
        with patch.object(rh_service, '_service_is_upstart', _m_bool()):
            assert rh_service.available('salt-api')

def test_missing():
    if False:
        return 10
    '\n    Test if it return True if the named service is not available.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool(False)):
        assert rh_service.missing('sshd', 'upstart')
        with patch.object(rh_service, '_service_is_sysv', _m_bool(False)):
            assert rh_service.missing('sshd')
    with patch.object(rh_service, '_service_is_sysv', _m_bool()):
        assert not rh_service.missing('sshd', 'sysvinit')
        with patch.object(rh_service, '_service_is_upstart', _m_bool()):
            assert not rh_service.missing('sshd')

def test_start():
    if False:
        while True:
            i = 10
    '\n    Test if it start the specified service.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        with patch.dict(rh_service.__salt__, {'cmd.retcode': _m_bool(False)}):
            assert rh_service.start('salt-api')

def test_stop():
    if False:
        while True:
            i = 10
    '\n    Test if it stop the specified service.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        with patch.dict(rh_service.__salt__, {'cmd.retcode': _m_bool(False)}):
            assert rh_service.stop('salt-api')

def test_restart():
    if False:
        while True:
            i = 10
    '\n    Test if it restart the specified service.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        with patch.dict(rh_service.__salt__, {'cmd.retcode': _m_bool(False)}):
            assert rh_service.restart('salt-api')

def test_reload():
    if False:
        return 10
    '\n    Test if it reload the specified service.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        with patch.dict(rh_service.__salt__, {'cmd.retcode': _m_bool(False)}):
            assert rh_service.reload_('salt-api')

def test_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return the status for a service,\n    returns a bool whether the service is running.\n    '
    with patch.object(rh_service, '_service_is_upstart', _m_bool()):
        mock_run = MagicMock(return_value='start/running')
        with patch.dict(rh_service.__salt__, {'cmd.run': mock_run}):
            assert rh_service.status('salt-api')
    with patch.object(rh_service, '_service_is_upstart', _m_bool(False)):
        with patch.dict(rh_service.__salt__, {'status.pid': _m_bool()}):
            assert rh_service.status('salt-api', sig=True)
        mock_ret = MagicMock(return_value=0)
        with patch.dict(rh_service.__salt__, {'cmd.retcode': mock_ret}):
            assert rh_service.status('salt-api')

def test_enable():
    if False:
        i = 10
        return i + 15
    '\n    Test if it enable the named service to start at boot.\n    '
    mock_bool = MagicMock(side_effect=[True, True, False])
    with patch.object(rh_service, '_service_is_upstart', mock_bool):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=True)):
            with patch.object(rh_service, '_upstart_enable', MagicMock(return_value=False)):
                assert not rh_service.enable('salt-api')
            with patch.object(rh_service, '_upstart_enable', MagicMock(return_value=True)):
                assert rh_service.enable('salt-api')
        with patch.object(rh_service, '_sysv_enable', _m_bool()):
            assert rh_service.enable('salt-api')

def test_disable():
    if False:
        print('Hello World!')
    '\n    Test if it disable the named service to start at boot.\n    '
    mock_bool = MagicMock(side_effect=[True, True, False])
    with patch.object(rh_service, '_service_is_upstart', mock_bool):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=True)):
            with patch.object(rh_service, '_upstart_disable', MagicMock(return_value=False)):
                assert not rh_service.disable('salt-api')
            with patch.object(rh_service, '_upstart_disable', MagicMock(return_value=True)):
                assert rh_service.disable('salt-api')
        with patch.object(rh_service, '_sysv_disable', _m_bool()):
            assert rh_service.disable('salt-api')

def test_enabled():
    if False:
        return 10
    '\n    Test if it check to see if the named service is enabled\n    to start on boot.\n    '
    mock_bool = MagicMock(side_effect=[True, False])
    with patch.object(rh_service, '_service_is_upstart', mock_bool):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=False)):
            assert not rh_service.enabled('salt-api')
        with patch.object(rh_service, '_sysv_is_enabled', _m_bool()):
            assert rh_service.enabled('salt-api')

def test_disabled():
    if False:
        while True:
            i = 10
    '\n    Test if it check to see if the named service is disabled\n    to start on boot.\n    '
    mock_bool = MagicMock(side_effect=[True, False])
    with patch.object(rh_service, '_service_is_upstart', mock_bool):
        with patch.object(rh_service, '_upstart_is_enabled', MagicMock(return_value=False)):
            assert rh_service.disabled('salt-api')
        with patch.object(rh_service, '_sysv_is_enabled', _m_bool(False)):
            assert rh_service.disabled('salt-api')