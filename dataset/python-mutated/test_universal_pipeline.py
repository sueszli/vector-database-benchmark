import logging
import pickle
try:
    from unittest import mock
except ImportError:
    import mock
import requests
import pytest
from azure.core.exceptions import DecodeError, AzureError
from azure.core.pipeline import Pipeline, PipelineResponse, PipelineRequest, PipelineContext
from azure.core.pipeline.policies import NetworkTraceLoggingPolicy, ContentDecodePolicy, RequestHistory, RetryPolicy, HTTPPolicy
from utils import HTTP_REQUESTS, create_http_request, HTTP_RESPONSES, REQUESTS_TRANSPORT_RESPONSES, create_http_response, create_transport_response, request_and_responses_product
from azure.core.pipeline._tools import is_rest

def test_pipeline_context():
    if False:
        while True:
            i = 10
    kwargs = {'stream': True, 'cont_token': 'bla'}
    context = PipelineContext('transport', **kwargs)
    context['foo'] = 'bar'
    context['xyz'] = '123'
    context['deserialized_data'] = 'marvelous'
    assert context['foo'] == 'bar'
    assert context.options == kwargs
    with pytest.raises(TypeError):
        context.clear()
    with pytest.raises(TypeError):
        context.update({})
    assert context.pop('foo') == 'bar'
    assert 'foo' not in context
    serialized = pickle.dumps(context)
    revived_context = pickle.loads(serialized)
    assert revived_context.options == kwargs
    assert revived_context.transport is None
    assert 'deserialized_data' in revived_context
    assert len(revived_context) == 1

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_request_history(http_request):
    if False:
        return 10

    class Non_deep_copyable(object):

        def __deepcopy__(self, memodict={}):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError()
    body = Non_deep_copyable()
    request = create_http_request(http_request, 'GET', 'http://localhost/', {'user-agent': 'test_request_history'})
    request.body = body
    request_history = RequestHistory(request)
    assert request_history.http_request.headers == request.headers
    assert request_history.http_request.url == request.url
    assert request_history.http_request.method == request.method

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_request_history_type_error(http_request):
    if False:
        for i in range(10):
            print('nop')

    class Non_deep_copyable(object):

        def __deepcopy__(self, memodict={}):
            if False:
                i = 10
                return i + 15
            raise TypeError()
    body = Non_deep_copyable()
    request = create_http_request(http_request, 'GET', 'http://localhost/', {'user-agent': 'test_request_history'})
    request.body = body
    request_history = RequestHistory(request)
    assert request_history.http_request.headers == request.headers
    assert request_history.http_request.url == request.url
    assert request_history.http_request.method == request.method

@mock.patch('azure.core.pipeline.policies._universal._LOGGER')
@pytest.mark.parametrize('http_request,http_response', request_and_responses_product(HTTP_RESPONSES))
def test_no_log(mock_http_logger, http_request, http_response):
    if False:
        print('Hello World!')
    universal_request = http_request('GET', 'http://localhost/')
    request = PipelineRequest(universal_request, PipelineContext(None))
    http_logger = NetworkTraceLoggingPolicy()
    response = PipelineResponse(request, create_http_response(http_response, universal_request, None), request.context)
    http_logger.on_request(request)
    mock_http_logger.debug.assert_not_called()
    http_logger.on_response(request, response)
    mock_http_logger.debug.assert_not_called()
    mock_http_logger.reset_mock()
    request.context.options['logging_enable'] = True
    http_logger.on_request(request)
    assert mock_http_logger.debug.call_count >= 1
    mock_http_logger.reset_mock()
    request.context.options['logging_enable'] = True
    http_logger.on_response(request, response)
    assert mock_http_logger.debug.call_count >= 1
    mock_http_logger.reset_mock()
    request.context.options['logging_enable'] = False
    http_logger.on_request(request)
    mock_http_logger.debug.assert_not_called()
    request.context.options['logging_enable'] = False
    http_logger.on_response(request, response)
    mock_http_logger.debug.assert_not_called()
    mock_http_logger.reset_mock()
    request.context.options = {}
    http_logger.enable_http_logger = True
    http_logger.on_request(request)
    assert mock_http_logger.debug.call_count >= 1
    http_logger.on_response(request, response)
    assert mock_http_logger.debug.call_count >= 1
    mock_http_logger.reset_mock()
    http_logger.enable_http_logger = True
    request.context.options['logging_enable'] = False
    http_logger.on_request(request)
    mock_http_logger.debug.assert_not_called()
    response.context['logging_enable'] = False
    http_logger.on_response(request, response)
    mock_http_logger.debug.assert_not_called()
    mock_http_logger.reset_mock()
    request.context.options['logging_enable'] = True
    http_logger.on_request(request)
    http_logger.on_response(request, response)
    first_count = mock_http_logger.debug.call_count
    assert first_count >= 1
    http_logger.on_request(request)
    http_logger.on_response(request, response)
    second_count = mock_http_logger.debug.call_count
    assert second_count == first_count * 2

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_retry_without_http_response(http_request):
    if False:
        for i in range(10):
            print('nop')

    class NaughtyPolicy(HTTPPolicy):

        def send(*args):
            if False:
                return 10
            raise AzureError('boo')
    policies = [RetryPolicy(), NaughtyPolicy()]
    pipeline = Pipeline(policies=policies, transport=None)
    with pytest.raises(AzureError):
        pipeline.run(http_request('GET', url='https://foo.bar'))

@pytest.mark.parametrize('http_request,http_response,requests_transport_response', request_and_responses_product(HTTP_RESPONSES, REQUESTS_TRANSPORT_RESPONSES))
def test_raw_deserializer(http_request, http_response, requests_transport_response):
    if False:
        return 10
    raw_deserializer = ContentDecodePolicy()
    context = PipelineContext(None, stream=False)
    universal_request = http_request('GET', 'http://localhost/')
    request = PipelineRequest(universal_request, context)

    def build_response(body, content_type=None):
        if False:
            while True:
                i = 10
        if is_rest(http_response):

            class MockResponse(http_response):

                def __init__(self, body, content_type):
                    if False:
                        for i in range(10):
                            print('nop')
                    super(MockResponse, self).__init__(request=None, internal_response=None, status_code=400, reason='Bad Request', content_type='application/json', headers={}, stream_download_generator=None)
                    self._body = body
                    self.content_type = content_type

                def body(self):
                    if False:
                        i = 10
                        return i + 15
                    return self._body

                def read(self):
                    if False:
                        print('Hello World!')
                    self._content = self._body
                    return self.content
        else:

            class MockResponse(http_response):

                def __init__(self, body, content_type):
                    if False:
                        for i in range(10):
                            print('nop')
                    super(MockResponse, self).__init__(None, None)
                    self._body = body
                    self.content_type = content_type

                def body(self):
                    if False:
                        return 10
                    return self._body
        return PipelineResponse(request, MockResponse(body, content_type), context)
    response = build_response(b'<groot/>', content_type='application/xml')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result.tag == 'groot'
    response = build_response(b'\xef\xbb\xbf<utf8groot/>', content_type='application/xml')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result.tag == 'utf8groot'
    response = build_response('<groot language="français"/>'.encode('utf-8'), content_type='application/xml')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result.attrib['language'] == 'français'
    response = build_response(b'{"ugly": true}', content_type='application/xml')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['ugly'] is True
    response = build_response(b'gibberish', content_type='application/xml')
    with pytest.raises(DecodeError) as err:
        raw_deserializer.on_response(request, response)
    assert err.value.response is response.http_response
    response = build_response(b'{{gibberish}}', content_type='application/xml')
    with pytest.raises(DecodeError) as err:
        raw_deserializer.on_response(request, response)
    assert err.value.response is response.http_response
    response = build_response(b'{"success": true}', content_type='application/json')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['success'] is True
    response = build_response(b'\xef\xbb\xbf{"success": true}', content_type='application/json')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['success'] is True
    response = build_response(b'{"success": true}', content_type='application/vnd.microsoft.appconfig.kv.v1+json')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['success'] is True
    response = build_response(b'{"success": true}', content_type='text/vnd.microsoft.appconfig.kv.v1+json')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['success'] is True
    response = build_response(b'"data"')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'data'
    response = build_response(b'I am groot', content_type='text/plain')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'I am groot'
    response = build_response(b'\xef\xbb\xbfI am groot', content_type='text/plain')
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'I am groot'
    req_response = requests.Response()
    req_response.headers['content-type'] = 'application/json'
    req_response._content = b'{"success": true}'
    req_response._content_consumed = True
    response = PipelineResponse(None, create_transport_response(requests_transport_response, None, req_response), PipelineContext(None, stream=False))
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result['success'] is True
    request.context.options['response_encoding'] = 'utf-8'
    response = build_response(b'\xc3\xa9', content_type='text/plain')
    raw_deserializer.on_request(request)
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'é'
    assert response.context['response_encoding'] == 'utf-8'
    del request.context['response_encoding']
    raw_deserializer = ContentDecodePolicy(response_encoding='utf-8')
    response = build_response(b'\xc3\xa9', content_type='text/plain')
    raw_deserializer.on_request(request)
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'é'
    assert response.context['response_encoding'] == 'utf-8'
    del request.context['response_encoding']
    request.context.options['response_encoding'] = 'utf-8-sig'
    response = build_response(b'\xc3\xa9', content_type='text/plain')
    raw_deserializer.on_request(request)
    raw_deserializer.on_response(request, response)
    result = response.context['deserialized_data']
    assert result == 'é'
    assert response.context['response_encoding'] == 'utf-8-sig'
    del request.context['response_encoding']

def test_json_merge_patch():
    if False:
        print('Hello World!')
    assert ContentDecodePolicy.deserialize_from_text('{"hello": "world"}', mime_type='application/merge-patch+json') == {'hello': 'world'}

def test_json_regex():
    if False:
        for i in range(10):
            print('nop')
    assert not ContentDecodePolicy.JSON_REGEXP.match('text/plain')
    assert ContentDecodePolicy.JSON_REGEXP.match('application/json')
    assert ContentDecodePolicy.JSON_REGEXP.match('text/json')
    assert ContentDecodePolicy.JSON_REGEXP.match('application/merge-patch+json')
    assert ContentDecodePolicy.JSON_REGEXP.match('application/ld+json')
    assert ContentDecodePolicy.JSON_REGEXP.match('application/vnd.microsoft.appconfig.kv+json')
    assert not ContentDecodePolicy.JSON_REGEXP.match('application/+json')
    assert not ContentDecodePolicy.JSON_REGEXP.match('application/not-json')
    assert not ContentDecodePolicy.JSON_REGEXP.match('application/iamjson')
    assert not ContentDecodePolicy.JSON_REGEXP.match('fake/json')