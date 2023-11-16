from __future__ import annotations
import logging
import ssl
from unittest import mock
import pytest
from docker import TLSConfig
from docker.errors import APIError
from airflow.exceptions import AirflowException, AirflowNotFoundException
from airflow.providers.docker.hooks.docker import DockerHook
TEST_CONN_ID = 'docker_test_connection'
TEST_BASE_URL = 'unix://var/run/docker.sock'
TEST_TLS_BASE_URL = 'tcp://localhost.foo.bar'
TEST_HTTPS_BASE_URL = 'https://localhost.foo.bar'
TEST_VERSION = '3.14'
TEST_CONN = {'host': 'some.docker.registry.com', 'login': 'some_user', 'password': 'some_p4$$w0rd'}
MOCK_CONNECTION_NOT_EXIST_MSG = 'Testing connection not exists'
MOCK_CONNECTION_NOT_EXISTS_EX = AirflowNotFoundException(MOCK_CONNECTION_NOT_EXIST_MSG)
HOOK_LOGGER_NAME = 'airflow.task.hooks.airflow.providers.docker.hooks.docker.DockerHook'

@pytest.fixture
def hook_kwargs():
    if False:
        return 10
    'Valid attributes for DockerHook.'
    return {'base_url': TEST_BASE_URL, 'docker_conn_id': 'docker_default', 'tls': False, 'version': TEST_VERSION, 'timeout': 42}

def test_no_connection_during_initialisation(hook_conn, docker_api_client_patcher, hook_kwargs):
    if False:
        print('Hello World!')
    "Hook shouldn't create client during initialisation and retrieve Airflow connection."
    DockerHook(**hook_kwargs)
    hook_conn.assert_not_called()
    docker_api_client_patcher.assert_not_called()

def test_init_fails_when_no_base_url_given(hook_kwargs):
    if False:
        while True:
            i = 10
    'Test mandatory `base_url` Hook argument.'
    hook_kwargs.pop('base_url')
    with pytest.raises(AirflowException, match='URL to the Docker server not provided\\.'):
        DockerHook(**hook_kwargs)

@pytest.mark.parametrize('base_url', ['http://foo.bar', TEST_BASE_URL])
@pytest.mark.parametrize('tls_config', [pytest.param(True, id='bool'), pytest.param(TLSConfig(), id='TLSConfig-object')])
def test_init_warn_on_non_https_host_with_enabled_tls(base_url, tls_config, hook_kwargs, caplog):
    if False:
        print('Hello World!')
    'Test warning if user specified tls but use non-https scheme.'
    caplog.set_level(logging.WARNING, logger=HOOK_LOGGER_NAME)
    hook_kwargs['base_url'] = base_url
    hook_kwargs['tls'] = tls_config
    DockerHook(**hook_kwargs)
    assert "When `tls` specified then `base_url` expected 'https://' schema." in caplog.messages

@pytest.mark.parametrize('hook_attr', ['docker_conn_id', 'version', 'tls', 'timeout'])
def test_optional_hook_attributes(hook_attr, hook_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Test if not provided optional arguments than Hook init nop failed.'
    hook_kwargs.pop(hook_attr)
    DockerHook(**hook_kwargs)

@pytest.mark.parametrize('conn_id, hook_conn', [pytest.param(TEST_CONN_ID, None, id='conn-specified'), pytest.param(None, MOCK_CONNECTION_NOT_EXISTS_EX, id='conn-not-specified')], indirect=['hook_conn'])
def test_create_api_client(conn_id, hook_conn, docker_api_client_patcher, caplog):
    if False:
        i = 10
        return i + 15
    "\n    Test creation ``docker.APIClient`` from hook arguments.\n    Additionally check:\n        - Is tls:// changed to https://\n        - Is ``api_client`` property and ``get_conn`` method cacheable.\n        - If `docker_conn_id` not provided that hook doesn't try access to Airflow Connections.\n    "
    caplog.set_level(logging.DEBUG, logger=HOOK_LOGGER_NAME)
    hook = DockerHook(docker_conn_id=conn_id, base_url=TEST_TLS_BASE_URL, version=TEST_VERSION, tls=True, timeout=42)
    assert "Change `base_url` schema from 'tcp://' to 'https://'." in caplog.messages
    caplog.clear()
    assert hook.client_created is False
    api_client = hook.api_client
    assert api_client is hook.get_conn(), 'Docker API Client not cacheable'
    docker_api_client_patcher.assert_called_once_with(base_url=TEST_HTTPS_BASE_URL, version=TEST_VERSION, tls=True, timeout=42)
    assert hook.client_created is True

def test_failed_create_api_client(docker_api_client_patcher):
    if False:
        return 10
    'Test failures during creation ``docker.APIClient`` from hook arguments.'
    hook = DockerHook(base_url=TEST_BASE_URL)
    docker_api_client_patcher.side_effect = Exception('Fake Exception')
    with pytest.raises(Exception, match='Fake Exception'):
        hook.get_conn()
    assert hook.client_created is False

@pytest.mark.parametrize('hook_conn, expected', [pytest.param(TEST_CONN, {'username': 'some_user', 'password': 'some_p4$$w0rd', 'registry': 'some.docker.registry.com', 'email': None, 'reauth': True}, id='host-login-password'), pytest.param({'host': 'another.docker.registry.com', 'login': 'another_user', 'password': 'insecure_password', 'extra': {'email': 'foo@bar.spam.egg', 'reauth': 'no'}}, {'username': 'another_user', 'password': 'insecure_password', 'registry': 'another.docker.registry.com', 'email': 'foo@bar.spam.egg', 'reauth': False}, id='host-login-password-email-noreauth'), pytest.param({'host': 'localhost', 'port': 8080, 'login': 'user', 'password': 'pass', 'extra': {'email': '', 'reauth': 'TrUe'}}, {'username': 'user', 'password': 'pass', 'registry': 'localhost:8080', 'email': None, 'reauth': True}, id='host-port-login-password-reauth')], indirect=['hook_conn'])
def test_success_login_to_registry(hook_conn, docker_api_client_patcher, expected: dict):
    if False:
        i = 10
        return i + 15
    'Test success login to Docker Registry with provided connection.'
    mock_login = mock.MagicMock()
    docker_api_client_patcher.return_value.login = mock_login
    hook = DockerHook(docker_conn_id=TEST_CONN_ID, base_url=TEST_BASE_URL)
    hook.get_conn()
    mock_login.assert_called_once_with(**expected)

def test_failed_login_to_registry(hook_conn, docker_api_client_patcher, caplog):
    if False:
        i = 10
        return i + 15
    'Test error during Docker Registry login.'
    caplog.set_level(logging.ERROR, logger=HOOK_LOGGER_NAME)
    docker_api_client_patcher.return_value.login.side_effect = APIError('Fake Error')
    hook = DockerHook(docker_conn_id=TEST_CONN_ID, base_url=TEST_BASE_URL)
    with pytest.raises(APIError, match='Fake Error'):
        hook.get_conn()
    assert 'Login failed' in caplog.messages

@pytest.mark.parametrize('hook_conn, ex, error_message', [pytest.param({k: v for (k, v) in TEST_CONN.items() if k != 'login'}, AirflowNotFoundException, 'No Docker Registry username provided\\.', id='missing-username'), pytest.param({k: v for (k, v) in TEST_CONN.items() if k != 'host'}, AirflowNotFoundException, 'No Docker Registry URL provided\\.', id='missing-registry-host'), pytest.param({**TEST_CONN, **{'extra': {'reauth': 'enabled'}}}, ValueError, "Unable parse `reauth` value '.*' to bool\\.", id='wrong-reauth'), pytest.param({**TEST_CONN, **{'extra': {'reauth': 'disabled'}}}, ValueError, "Unable parse `reauth` value '.*' to bool\\.", id='wrong-noreauth')], indirect=['hook_conn'])
def test_invalid_conn_parameters(hook_conn, docker_api_client_patcher, ex, error_message):
    if False:
        for i in range(10):
            print('nop')
    'Test invalid/missing connection parameters.'
    hook = DockerHook(docker_conn_id=TEST_CONN_ID, base_url=TEST_BASE_URL)
    with pytest.raises(ex, match=error_message):
        hook.get_conn()

@pytest.mark.parametrize('tls_params', [pytest.param({}, id='empty-config'), pytest.param({'client_cert': 'foo-bar', 'client_key': 'spam-egg'}, id='missing-ca-cert'), pytest.param({'ca_cert': 'foo-bar', 'client_key': 'spam-egg'}, id='missing-client-cert'), pytest.param({'ca_cert': 'foo-bar', 'client_cert': 'spam-egg'}, id='missing-client-key')])
def test_construct_tls_config_missing_certs_args(tls_params: dict):
    if False:
        return 10
    'Test that return False on missing cert/keys arguments.'
    assert DockerHook.construct_tls_config(**tls_params) is False

@pytest.mark.parametrize('assert_hostname', ['foo.bar', None, False])
@pytest.mark.parametrize('ssl_version', [pytest.param(ssl.PROTOCOL_TLSv1, id='TLSv1'), pytest.param(ssl.PROTOCOL_TLSv1_2, id='TLSv1_2'), None])
def test_construct_tls_config(assert_hostname, ssl_version):
    if False:
        for i in range(10):
            print('nop')
    'Test construct ``docker.tls.TLSConfig`` object.'
    tls_params = {'ca_cert': 'test-ca', 'client_cert': 'foo-bar', 'client_key': 'spam-egg'}
    expected_call_args = {'ca_cert': 'test-ca', 'client_cert': ('foo-bar', 'spam-egg'), 'verify': True}
    if assert_hostname is not None:
        tls_params['assert_hostname'] = assert_hostname
    if ssl_version is not None:
        tls_params['ssl_version'] = ssl_version
    with mock.patch.object(TLSConfig, '__init__', return_value=None) as mock_tls_config:
        DockerHook.construct_tls_config(**tls_params)
        mock_tls_config.assert_called_once_with(**expected_call_args, assert_hostname=assert_hostname, ssl_version=ssl_version)