"""
tests.unit.proxy.test_cimc
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for the cimc proxy module
"""
import logging
import pytest
import salt.exceptions
import salt.proxy.cimc as cimc
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)

def http_query_response(*args, data=None, **kwargs):
    if False:
        while True:
            i = 10
    log.debug('http_query_response side_effect; ARGS: %s // KWARGS: %s // DATA: %s', args, kwargs, data)
    login_response = '    <aaaLogin\n        response="yes"\n        outCookie="real-cookie"\n        outRefreshPeriod="600"\n        outPriv="admin">\n    </aaaLogin>'
    logout_response = '    <aaaLogout\n        cookie="real-cookie"\n        response="yes"\n        outStatus="success">\n    </aaaLogout>\n    '
    config_resolve_class_response = '    <configResolveClass\n        cookie="real-cookie"\n        response="yes"\n        classId="computeRackUnit">\n        <outConfig>\n            <computeRackUnit\n                dn="sys/rack-unit-1"\n                adminPower="policy"\n                availableMemory="16384"\n                model="R210-2121605W"\n                memorySpeed="1067"\n                name="UCS C210 M2"\n                numOfAdaptors="2"\n                numOfCores="8"\n                numOfCoresEnabled="8"\n                numOfCpus="2"\n                numOfEthHostIfs="5"\n                numOfFcHostIfs="2"\n                numOfThreads="16"\n                operPower="on"\n                originalUuid="00C9DE3C-370D-DF11-1186-6DD1393A608B"\n                presence="equipped"\n                serverID="1"\n                serial="QCI140205Z2"\n                totalMemory="16384"\n                usrLbl="C210 Row-B Rack-10"\n                uuid="00C9DE3C-370D-DF11-1186-6DD1393A608B"\n                vendor="Cisco Systems Inc" >\n            </computeRackUnit>\n        </outConfig>\n    </configResolveClass>\n    '
    config_con_mo_response = '    <configConfMo\n        dn="sys/rack-unit-1/locator-led"\n        cookie="real-cookie"\n        response="yes">\n        <outConfig>\n            <equipmentLocatorLed\n                dn="sys/rack-unit-1/locator-led"\n                adminState="inactive"\n                color="unknown"\n                id="1"\n                name=""\n                operState="off">\n            </equipmentLocatorLed>\n        </outConfig>\n    </configConfMo>\n    '
    if data.startswith('<aaaLogin'):
        response = login_response
    elif data.startswith('<aaaLogout'):
        response = logout_response
    elif data.startswith('<configResolveClass'):
        response = config_resolve_class_response
    elif data.startswith('<configConfMo'):
        response = config_con_mo_response
    else:
        response = ''
    return {'text': response, 'status': 200}

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    with patch.dict(cimc.DETAILS):
        yield {cimc: {'__pillar__': {}}}

@pytest.fixture(params=[None, True, False])
def verify_ssl(request):
    if False:
        return 10
    return request.param

@pytest.fixture
def opts(verify_ssl):
    if False:
        print('Hello World!')
    return {'proxy': {'host': 'TheHost', 'username': 'TheUsername', 'password': 'ThePassword', 'verify_ssl': verify_ssl}}

def _get_expected_verify_ssl(verify_ssl):
    if False:
        print('Hello World!')
    expected = True if verify_ssl is None else verify_ssl
    log.debug('verify_ssl: %s // expected verify_ssl: %s', verify_ssl, expected)
    return expected

def test_init():
    if False:
        while True:
            i = 10
    opts = {'proxy': {'username': 'xxxx', 'password': 'xxx'}}
    ret = cimc.init(opts)
    assert not ret
    opts = {'proxy': {'password': 'xxx', 'host': 'cimc'}}
    ret = cimc.init(opts)
    assert not ret
    opts = {'proxy': {'username': 'xxxx', 'host': 'cimc'}}
    ret = cimc.init(opts)
    assert not ret
    opts = {'proxy': {'username': 'xxxx', 'password': 'xxx', 'host': 'cimc'}}
    with patch.object(cimc, 'logon', return_value='9zVG5U8DFZNsTR'):
        with patch.object(cimc, 'get_config_resolver_class', return_value='True'):
            ret = cimc.init(opts)
            assert cimc.DETAILS['url'] == 'https://cimc/nuova'
            assert cimc.DETAILS['username'] == 'xxxx'
            assert cimc.DETAILS['password'] == 'xxx'
            assert cimc.DETAILS['initialized']

def test__validate_response_code():
    if False:
        i = 10
        return i + 15
    with pytest.raises(salt.exceptions.CommandExecutionError, match='Did not receive a valid response from host.'):
        cimc._validate_response_code('404')
    with patch.object(cimc, 'logout', return_value=True) as mock_logout:
        with pytest.raises(salt.exceptions.CommandExecutionError, match='Did not receive a valid response from host.'):
            cimc._validate_response_code('404', '9zVG5U8DFZNsTR')
            mock_logout.assert_called_once_with('9zVG5U8DFZNsTR')

def test_init_with_ssl(verify_ssl, opts):
    if False:
        print('Hello World!')
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.init(opts)
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_logon(opts, verify_ssl):
    if False:
        return 10
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.logon()
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_logout(opts, verify_ssl):
    if False:
        for i in range(10):
            print('nop')
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.logout()
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_grains(opts, verify_ssl):
    if False:
        for i in range(10):
            print('nop')
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.grains()
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_grains_refresh(opts, verify_ssl):
    if False:
        for i in range(10):
            print('nop')
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.grains_refresh()
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_ping(opts, verify_ssl):
    if False:
        i = 10
        return i + 15
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.ping()
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error

def test_set_config_modify(opts, verify_ssl):
    if False:
        while True:
            i = 10
    http_query_mock = MagicMock(side_effect=http_query_response)
    expected_verify_ssl_value = _get_expected_verify_ssl(verify_ssl)
    with patch('salt.proxy.cimc.get_config_resolver_class', MagicMock(return_value=True)):
        cimc.init(opts)
    with patch.dict(cimc.__utils__, {'http.query': http_query_mock}):
        cimc.set_config_modify(dn='sys/rack-unit-1/locator-led', inconfig="<inConfig><equipmentLocatorLed adminState='on' dn='sys/rack-unit-1/locator-led'></equipmentLocatorLed></inConfig>")
    for (idx, call) in enumerate(http_query_mock.mock_calls, 1):
        condition = call.kwargs['verify_ssl'] is expected_verify_ssl_value
        condition_error = '{} != {}; Call(number={}): {}'.format(idx, call, call.kwargs['verify_ssl'], expected_verify_ssl_value)
        assert condition, condition_error