from __future__ import annotations
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import Connection
from airflow.providers.openfaas.hooks.openfaas import OpenFaasHook
FUNCTION_NAME = 'function_name'

class TestOpenFaasHook:
    GET_FUNCTION = '/system/function/'
    INVOKE_ASYNC_FUNCTION = '/async-function/'
    INVOKE_FUNCTION = '/function/'
    DEPLOY_FUNCTION = '/system/functions'
    UPDATE_FUNCTION = '/system/functions'

    def setup_method(self):
        if False:
            return 10
        self.hook = OpenFaasHook(function_name=FUNCTION_NAME)
        self.mock_response = {'ans': 'a'}

    @mock.patch.object(BaseHook, 'get_connection')
    def test_is_function_exist_false(self, mock_get_connection, requests_mock):
        if False:
            i = 10
            return i + 15
        requests_mock.get('http://open-faas.io' + self.GET_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=404)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        does_function_exist = self.hook.does_function_exist()
        assert not does_function_exist

    @mock.patch.object(BaseHook, 'get_connection')
    def test_is_function_exist_true(self, mock_get_connection, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.get('http://open-faas.io' + self.GET_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=202)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        does_function_exist = self.hook.does_function_exist()
        assert does_function_exist

    @mock.patch.object(BaseHook, 'get_connection')
    def test_update_function_true(self, mock_get_connection, requests_mock):
        if False:
            while True:
                i = 10
        requests_mock.put('http://open-faas.io' + self.UPDATE_FUNCTION, json=self.mock_response, status_code=202)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        self.hook.update_function({})

    @mock.patch.object(BaseHook, 'get_connection')
    def test_update_function_false(self, mock_get_connection, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.put('http://open-faas.io' + self.UPDATE_FUNCTION, json=self.mock_response, status_code=400)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        with pytest.raises(AirflowException) as ctx:
            self.hook.update_function({})
        assert 'failed to update ' + FUNCTION_NAME in str(ctx.value)

    @mock.patch.object(BaseHook, 'get_connection')
    def test_invoke_function_false(self, mock_get_connection, requests_mock):
        if False:
            return 10
        requests_mock.post('http://open-faas.io' + self.INVOKE_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=400)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        with pytest.raises(AirflowException) as ctx:
            self.hook.invoke_function({})
        assert 'failed to invoke function' in str(ctx.value)

    @mock.patch.object(BaseHook, 'get_connection')
    def test_invoke_function_true(self, mock_get_connection, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.post('http://open-faas.io' + self.INVOKE_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=200)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        assert self.hook.invoke_function({}) is None

    @mock.patch.object(BaseHook, 'get_connection')
    def test_invoke_async_function_false(self, mock_get_connection, requests_mock):
        if False:
            print('Hello World!')
        requests_mock.post('http://open-faas.io' + self.INVOKE_ASYNC_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=400)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        with pytest.raises(AirflowException) as ctx:
            self.hook.invoke_async_function({})
        assert 'failed to invoke function' in str(ctx.value)

    @mock.patch.object(BaseHook, 'get_connection')
    def test_invoke_async_function_true(self, mock_get_connection, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.post('http://open-faas.io' + self.INVOKE_ASYNC_FUNCTION + FUNCTION_NAME, json=self.mock_response, status_code=202)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        assert self.hook.invoke_async_function({}) is None

    @mock.patch.object(BaseHook, 'get_connection')
    def test_deploy_function_function_already_exist(self, mock_get_connection, requests_mock):
        if False:
            for i in range(10):
                print('nop')
        requests_mock.put('http://open-faas.io/' + self.UPDATE_FUNCTION, json=self.mock_response, status_code=202)
        mock_connection = Connection(host='http://open-faas.io/')
        mock_get_connection.return_value = mock_connection
        assert self.hook.deploy_function(True, {}) is None

    @mock.patch.object(BaseHook, 'get_connection')
    def test_deploy_function_function_not_exist(self, mock_get_connection, requests_mock):
        if False:
            return 10
        requests_mock.post('http://open-faas.io' + self.DEPLOY_FUNCTION, json={}, status_code=202)
        mock_connection = Connection(host='http://open-faas.io')
        mock_get_connection.return_value = mock_connection
        assert self.hook.deploy_function(False, {}) is None