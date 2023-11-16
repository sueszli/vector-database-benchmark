from __future__ import annotations
import os
from unittest import mock
from unittest.mock import MagicMock, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.yandex.hooks.yandex import YandexCloudBaseHook
from tests.test_utils.config import conf_vars

class TestYandexHook:

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials')
    def test_client_created_without_exceptions(self, get_credentials_mock, get_connection_mock):
        if False:
            for i in range(10):
                print('nop')
        'tests `init` method to validate client creation when all parameters are passed'
        default_folder_id = 'test_id'
        default_public_ssh_key = 'test_key'
        extra_dejson = '{"extras": "extra"}'
        get_connection_mock['extra_dejson'] = 'sdsd'
        get_connection_mock.extra_dejson = '{"extras": "extra"}'
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson=extra_dejson)
        get_credentials_mock.return_value = {'token': 122323}
        hook = YandexCloudBaseHook(yandex_conn_id=None, default_folder_id=default_folder_id, default_public_ssh_key=default_public_ssh_key)
        assert hook.client is not None

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    def test_get_credentials_raise_exception(self, get_connection_mock):
        if False:
            for i in range(10):
                print('nop')
        "tests 'get_credentials' method raising exception if none of the required fields are passed."
        default_folder_id = 'test_id'
        default_public_ssh_key = 'test_key'
        extra_dejson = '{"extras": "extra"}'
        get_connection_mock['extra_dejson'] = 'sdsd'
        get_connection_mock.extra_dejson = '{"extras": "extra"}'
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson=extra_dejson)
        with pytest.raises(AirflowException):
            YandexCloudBaseHook(yandex_conn_id=None, default_folder_id=default_folder_id, default_public_ssh_key=default_public_ssh_key)

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials')
    def test_get_field(self, get_credentials_mock, get_connection_mock):
        if False:
            for i in range(10):
                print('nop')
        default_folder_id = 'test_id'
        default_public_ssh_key = 'test_key'
        extra_dejson = {'one': 'value_one'}
        get_connection_mock['extra_dejson'] = 'sdsd'
        get_connection_mock.extra_dejson = '{"extras": "extra"}'
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson=extra_dejson)
        get_credentials_mock.return_value = {'token': 122323}
        hook = YandexCloudBaseHook(yandex_conn_id=None, default_folder_id=default_folder_id, default_public_ssh_key=default_public_ssh_key)
        assert hook._get_field('one') == 'value_one'

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials')
    def test_get_endpoint_specified(self, get_credentials_mock, get_connection_mock):
        if False:
            print('Hello World!')
        default_folder_id = 'test_id'
        default_public_ssh_key = 'test_key'
        extra_dejson = {'endpoint': 'my_endpoint', 'something_else': 'some_value'}
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson=extra_dejson)
        get_credentials_mock.return_value = {'token': 122323}
        hook = YandexCloudBaseHook(yandex_conn_id=None, default_folder_id=default_folder_id, default_public_ssh_key=default_public_ssh_key)
        assert hook._get_endpoint() == {'endpoint': 'my_endpoint'}

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials')
    def test_get_endpoint_unspecified(self, get_credentials_mock, get_connection_mock):
        if False:
            return 10
        default_folder_id = 'test_id'
        default_public_ssh_key = 'test_key'
        extra_dejson = {'something_else': 'some_value'}
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson=extra_dejson)
        get_credentials_mock.return_value = {'token': 122323}
        hook = YandexCloudBaseHook(yandex_conn_id=None, default_folder_id=default_folder_id, default_public_ssh_key=default_public_ssh_key)
        assert hook._get_endpoint() == {}

    @mock.patch('airflow.hooks.base.BaseHook.get_connection')
    @mock.patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials')
    def test_sdk_user_agent(self, get_credentials_mock, get_connection_mock):
        if False:
            return 10
        get_connection_mock.return_value = mock.Mock(connection_id='yandexcloud_default', extra_dejson='{}')
        get_credentials_mock.return_value = {'token': 122323}
        sdk_prefix = 'MyAirflow'
        with conf_vars({('yandex', 'sdk_user_agent_prefix'): sdk_prefix}):
            hook = YandexCloudBaseHook()
            assert hook.sdk._channels._client_user_agent.startswith(sdk_prefix)

    @pytest.mark.parametrize('uri', [pytest.param('a://?extra__yandexcloud__folder_id=abc&extra__yandexcloud__public_ssh_key=abc', id='prefix'), pytest.param('a://?folder_id=abc&public_ssh_key=abc', id='no-prefix')])
    @patch('airflow.providers.yandex.hooks.yandex.YandexCloudBaseHook._get_credentials', new=MagicMock())
    def test_backcompat_prefix_works(self, uri):
        if False:
            for i in range(10):
                print('nop')
        with patch.dict(os.environ, {'AIRFLOW_CONN_MY_CONN': uri}):
            hook = YandexCloudBaseHook('my_conn')
            assert hook.default_folder_id == 'abc'
            assert hook.default_public_ssh_key == 'abc'