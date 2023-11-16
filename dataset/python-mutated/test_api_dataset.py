import json
import socket
import pytest
import requests
import requests_mock
from kedro.extras.datasets.api import APIDataSet
from kedro.io.core import DatasetError
POSSIBLE_METHODS = ['GET', 'OPTIONS', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE']
TEST_URL = 'http://example.com/api/test'
TEST_TEXT_RESPONSE_DATA = 'This is a response.'
TEST_JSON_RESPONSE_DATA = [{'key': 'value'}]
TEST_PARAMS = {'param': 'value'}
TEST_URL_WITH_PARAMS = TEST_URL + '?param=value'
TEST_HEADERS = {'key': 'value'}

@pytest.mark.parametrize('method', POSSIBLE_METHODS)
class TestAPIDataSet:

    @pytest.fixture
    def requests_mocker(self):
        if False:
            print('Hello World!')
        with requests_mock.Mocker() as mock:
            yield mock

    def test_successfully_load_with_response(self, requests_mocker, method):
        if False:
            return 10
        api_data_set = APIDataSet(url=TEST_URL, method=method, params=TEST_PARAMS, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, headers=TEST_HEADERS, text=TEST_TEXT_RESPONSE_DATA)
        response = api_data_set.load()
        assert isinstance(response, requests.Response)
        assert response.text == TEST_TEXT_RESPONSE_DATA

    def test_successful_json_load_with_response(self, requests_mocker, method):
        if False:
            print('Hello World!')
        api_data_set = APIDataSet(url=TEST_URL, method=method, json=TEST_JSON_RESPONSE_DATA, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL, headers=TEST_HEADERS, text=json.dumps(TEST_JSON_RESPONSE_DATA))
        response = api_data_set.load()
        assert isinstance(response, requests.Response)
        assert response.json() == TEST_JSON_RESPONSE_DATA

    def test_http_error(self, requests_mocker, method):
        if False:
            while True:
                i = 10
        api_data_set = APIDataSet(url=TEST_URL, method=method, params=TEST_PARAMS, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, headers=TEST_HEADERS, text='Nope, not found', status_code=requests.codes.FORBIDDEN)
        with pytest.raises(DatasetError, match='Failed to fetch data'):
            api_data_set.load()

    def test_socket_error(self, requests_mocker, method):
        if False:
            while True:
                i = 10
        api_data_set = APIDataSet(url=TEST_URL, method=method, params=TEST_PARAMS, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, exc=socket.error)
        with pytest.raises(DatasetError, match='Failed to connect'):
            api_data_set.load()

    def test_read_only_mode(self, method):
        if False:
            i = 10
            return i + 15
        '\n        Saving is disabled on the data set.\n        '
        api_data_set = APIDataSet(url=TEST_URL, method=method)
        with pytest.raises(DatasetError, match='is a read only data set type'):
            api_data_set.save({})

    def test_exists_http_error(self, requests_mocker, method):
        if False:
            return 10
        '\n        In case of an unexpected HTTP error,\n        ``exists()`` should not silently catch it.\n        '
        api_data_set = APIDataSet(url=TEST_URL, method=method, params=TEST_PARAMS, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, headers=TEST_HEADERS, text='Nope, not found', status_code=requests.codes.FORBIDDEN)
        with pytest.raises(DatasetError, match='Failed to fetch data'):
            api_data_set.exists()

    def test_exists_ok(self, requests_mocker, method):
        if False:
            while True:
                i = 10
        '\n        If the file actually exists and server responds 200,\n        ``exists()`` should return True\n        '
        api_data_set = APIDataSet(url=TEST_URL, method=method, params=TEST_PARAMS, headers=TEST_HEADERS)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, headers=TEST_HEADERS, text=TEST_TEXT_RESPONSE_DATA)
        assert api_data_set.exists()

    def test_credentials_auth_error(self, method):
        if False:
            for i in range(10):
                print('nop')
        '\n        If ``auth`` and ``credentials`` are both provided,\n        the constructor should raise a ValueError.\n        '
        with pytest.raises(ValueError, match='both auth and credentials'):
            APIDataSet(url=TEST_URL, method=method, auth=[], credentials=[])

    @pytest.mark.parametrize('auth_kwarg', ['auth', 'credentials'])
    @pytest.mark.parametrize('auth_seq', [('username', 'password'), ['username', 'password'], (e for e in ['username', 'password'])])
    def test_auth_sequence(self, requests_mocker, method, auth_seq, auth_kwarg):
        if False:
            for i in range(10):
                print('nop')
        '\n        ``auth`` and ``credentials`` should be able to be any Iterable.\n        '
        kwargs = {'url': TEST_URL, 'method': method, 'params': TEST_PARAMS, 'headers': TEST_HEADERS, auth_kwarg: auth_seq}
        api_data_set = APIDataSet(**kwargs)
        requests_mocker.register_uri(method, TEST_URL_WITH_PARAMS, headers=TEST_HEADERS, text=TEST_TEXT_RESPONSE_DATA)
        response = api_data_set.load()
        assert isinstance(response, requests.Response)
        assert response.text == TEST_TEXT_RESPONSE_DATA