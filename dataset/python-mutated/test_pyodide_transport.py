"""Tests that mock the browser layer."""
import asyncio
import sys
from typing import NamedTuple
from unittest import mock
import pytest
from azure.core.exceptions import HttpResponseError
from azure.core.pipeline._base_async import AsyncPipeline
from azure.core.pipeline.policies._retry_async import AsyncRetryPolicy
from azure.core.rest import HttpRequest
PLACEHOLDER_ENDPOINT = 'https://my-resource-group.cognitiveservices.azure.com/'

class TestPyodideTransportClass:
    """Unittest for the Pyodide transport."""

    @pytest.fixture()
    def mock_pyodide_module(self):
        if False:
            return 10
        'Create a mock for the Pyodide module.'
        mock_pyodide_module = mock.Mock()
        mock_pyodide_module.http.pyfetch = mock.Mock()
        mock_pyodide_module.JsException = type('JsException', (Exception,), {})
        return mock_pyodide_module

    @pytest.fixture()
    def mock_js_module(self):
        if False:
            while True:
                i = 10
        'Mock the `js` module'
        return mock.Mock()

    @pytest.fixture()
    def transport(self, mock_pyodide_module, mock_js_module):
        if False:
            for i in range(10):
                print('nop')
        'Add the mock Pyodide module to `sys.modules` and import our transport.'
        patch_dict = (('pyodide', mock_pyodide_module), ('pyodide.http', mock_pyodide_module.http), ('js', mock_js_module))
        with mock.patch.dict(sys.modules, patch_dict):
            import azure.core.experimental.transport
            yield azure.core.experimental.transport

    @pytest.fixture()
    def pipeline(self, transport):
        if False:
            while True:
                i = 10
        'Create a pipeline to test.'
        return AsyncPipeline(transport.PyodideTransport(), [AsyncRetryPolicy()])

    @pytest.fixture()
    def mock_pyfetch(self, mock_pyodide_module):
        if False:
            return 10
        'Utility fixture for less typing.'
        return mock_pyodide_module.http.pyfetch

    def create_mock_response(self, body: bytes, headers: dict, status: int, status_text: str) -> mock.Mock:
        if False:
            print('Hello World!')
        'Create a mock response object that mimics `pyodide.http.FetchResponse`'
        mock_response = mock.Mock()
        mock_response.body = body
        mock_response.js_response.headers = headers
        mock_response.status = status
        mock_response.status_text = status_text
        bytes_promise = asyncio.Future()
        bytes_promise.set_result(body)
        mock_response.bytes = mock.Mock()
        mock_response.bytes.return_value = bytes_promise
        mock_response.clone.return_value = mock_response
        response_promise = asyncio.Future()
        response_promise.set_result(mock_response)
        return response_promise

    @pytest.mark.skipif(sys.version_info < (3, 8), reason='pyodide needs py 3.8+')
    @pytest.mark.asyncio
    async def test_successful_send(self, mock_pyfetch, mock_pyodide_module, pipeline):
        """Test that a successful send returns the correct values."""
        mock_pyfetch.reset_mock()
        method = 'POST'
        headers = {'key': 'value'}
        data = b'data'
        request = HttpRequest(method=method, url=PLACEHOLDER_ENDPOINT, headers=headers, data=data)
        response_body = b'0123'
        response_headers = {'header': 'value'}
        response_status = 200
        response_text = 'OK'
        mock_response = self.create_mock_response(body=response_body, headers=response_headers, status=response_status, status_text=response_text)
        mock_pyodide_module.http.pyfetch.return_value = mock_response
        response = (await pipeline.run(request=request)).http_response
        await response.load_body()
        assert response.body() == response_body
        assert response.status_code == response_status
        assert response.headers == response_headers
        assert response.reason == response_text
        assert not response._is_closed
        await response.close()
        assert response._is_closed
        mock_pyfetch.assert_called_once()
        args = mock_pyfetch.call_args[0]
        kwargs = mock_pyfetch.call_args[1]
        assert len(args) == 1
        assert args[0] == PLACEHOLDER_ENDPOINT
        assert kwargs['method'] == method
        assert kwargs['body'] == data
        assert not kwargs['allow_redirects']
        assert kwargs['headers']['key'] == 'value'
        assert kwargs['headers']['Content-Length'] == str(len(data))
        assert kwargs['verify']
        assert kwargs['cert'] is None
        assert not kwargs['files']

    @pytest.mark.skipif(sys.version_info < (3, 8), reason='pyodide needs py 3.8+')
    @pytest.mark.asyncio
    async def test_unsuccessful_send(self, mock_pyfetch, mock_pyodide_module, pipeline):
        """Test that the pipeline is failing correctly."""
        mock_pyfetch.reset_mock()
        mock_pyfetch.side_effect = mock_pyodide_module.JsException
        retry_total = 3
        request = HttpRequest(method='GET', url=PLACEHOLDER_ENDPOINT)
        with pytest.raises(HttpResponseError):
            await pipeline.run(request)
        assert mock_pyfetch.call_count == retry_total + 1

    @pytest.mark.skipif(sys.version_info < (3, 8), reason='pyodide needs py 3.8+')
    def test_valid_import(self, transport):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can import Pyodide classes from `azure.core.pipeline.transport`\n        Adding the transport fixture will mock the Pyodide modules in `sys.modules`.\n        '
        import azure.core.experimental.transport as transport
        assert transport.PyodideTransport