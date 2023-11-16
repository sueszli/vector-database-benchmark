import io
import pytest
import sys
import os
try:
    import collections.abc as collections
except ImportError:
    import collections
from azure.core.configuration import Configuration
from azure.core.rest import HttpRequest
from azure.core.pipeline.policies import CustomHookPolicy, UserAgentPolicy, SansIOHTTPPolicy, RetryPolicy
from azure.core.pipeline._tools import is_rest
from rest_client import MockRestClient
from azure.core import PipelineClient

@pytest.fixture
def assert_iterator_body():
    if False:
        return 10

    def _comparer(request, final_value):
        if False:
            i = 10
            return i + 15
        content = b''.join([p for p in request.content])
        assert content == final_value
    return _comparer

def test_request_repr():
    if False:
        i = 10
        return i + 15
    request = HttpRequest('GET', 'http://example.org')
    assert repr(request) == "<HttpRequest [GET], url: 'http://example.org'>"

def test_no_content():
    if False:
        print('Hello World!')
    request = HttpRequest('GET', 'http://example.org')
    assert 'Content-Length' not in request.headers

def test_content_length_header():
    if False:
        for i in range(10):
            print('nop')
    request = HttpRequest('POST', 'http://example.org', content=b'test 123')
    assert request.headers['Content-Length'] == '8'

def test_iterable_content(assert_iterator_body):
    if False:
        return 10

    class Content:

        def __iter__(self):
            if False:
                while True:
                    i = 10
            yield b'test 123'
    request = HttpRequest('POST', 'http://example.org', content=Content())
    assert request.headers == {}
    assert_iterator_body(request, b'test 123')

def test_generator_with_transfer_encoding_header(assert_iterator_body):
    if False:
        for i in range(10):
            print('nop')

    def content():
        if False:
            while True:
                i = 10
        yield b'test 123'
    request = HttpRequest('POST', 'http://example.org', content=content())
    assert request.headers == {}
    assert_iterator_body(request, b'test 123')

def test_generator_with_content_length_header(assert_iterator_body):
    if False:
        return 10

    def content():
        if False:
            i = 10
            return i + 15
        yield b'test 123'
    headers = {'Content-Length': '8'}
    request = HttpRequest('POST', 'http://example.org', content=content(), headers=headers)
    assert request.headers == {'Content-Length': '8'}
    assert_iterator_body(request, b'test 123')

def test_url_encoded_data():
    if False:
        while True:
            i = 10
    request = HttpRequest('POST', 'http://example.org', data={'test': '123'})
    assert request.headers['Content-Type'] == 'application/x-www-form-urlencoded'
    assert request.content == {'test': '123'}

def test_json_encoded_data():
    if False:
        i = 10
        return i + 15
    request = HttpRequest('POST', 'http://example.org', json={'test': 123})
    assert request.headers['Content-Type'] == 'application/json'
    assert request.content == '{"test": 123}'

def test_headers():
    if False:
        return 10
    request = HttpRequest('POST', 'http://example.org', json={'test': 123})
    assert request.headers == {'Content-Type': 'application/json', 'Content-Length': '13'}

def test_ignore_transfer_encoding_header_if_content_length_exists():
    if False:
        return 10
    '\n    `Transfer-Encoding` should be ignored if `Content-Length` has been set explicitly.\n    See https://github.com/encode/httpx/issues/1168\n    '

    def streaming_body(data):
        if False:
            print('Hello World!')
        yield data
    data = streaming_body(b'abcd')
    headers = {'Content-Length': '4'}
    request = HttpRequest('POST', 'http://example.org', data=data, headers=headers)
    assert 'Transfer-Encoding' not in request.headers
    assert request.headers['Content-Length'] == '4'

def test_override_accept_encoding_header():
    if False:
        print('Hello World!')
    headers = {'Accept-Encoding': 'identity'}
    request = HttpRequest('GET', 'http://example.org', headers=headers)
    assert request.headers['Accept-Encoding'] == 'identity'
'Test request body'

def test_empty_content():
    if False:
        i = 10
        return i + 15
    request = HttpRequest('GET', 'http://example.org')
    assert request.content is None

def test_string_content():
    if False:
        print('Hello World!')
    request = HttpRequest('PUT', 'http://example.org', content='Hello, world!')
    assert request.headers == {'Content-Length': '13', 'Content-Type': 'text/plain'}
    assert request.content == 'Hello, world!'
    request = HttpRequest('PUT', 'http://example.org', data='Hello, world!')
    assert request.headers == {'Content-Length': '13', 'Content-Type': 'text/plain'}
    assert request.content == 'Hello, world!'
    request = HttpRequest('GET', 'http://example.org', data='Hello, world!')
    assert request.headers == {'Content-Length': '13', 'Content-Type': 'text/plain'}
    assert request.content == 'Hello, world!'

def test_bytes_content():
    if False:
        print('Hello World!')
    request = HttpRequest('PUT', 'http://example.org', content=b'Hello, world!')
    assert request.headers == {'Content-Length': '13'}
    assert request.content == b'Hello, world!'
    request = HttpRequest('PUT', 'http://example.org', data=b'Hello, world!')
    assert request.headers == {'Content-Length': '13'}
    assert request.content == b'Hello, world!'
    request = HttpRequest('GET', 'http://example.org', data=b'Hello, world!')
    assert request.headers == {'Content-Length': '13'}
    assert request.content == b'Hello, world!'

def test_iterator_content(assert_iterator_body):
    if False:
        print('Hello World!')

    def hello_world():
        if False:
            i = 10
            return i + 15
        yield b'Hello, '
        yield b'world!'
    request = HttpRequest('POST', url='http://example.org', content=hello_world())
    assert isinstance(request.content, collections.Iterable)
    assert_iterator_body(request, b'Hello, world!')
    assert request.headers == {}
    request = HttpRequest('POST', url='http://example.org', data=hello_world())
    assert isinstance(request.content, collections.Iterable)
    assert_iterator_body(request, b'Hello, world!')
    assert request.headers == {}
    request = HttpRequest('GET', url='http://example.org', data=hello_world())
    assert isinstance(request.content, collections.Iterable)
    assert_iterator_body(request, b'Hello, world!')
    assert request.headers == {}

def test_json_content():
    if False:
        return 10
    request = HttpRequest('POST', url='http://example.org', json={'Hello': 'world!'})
    assert request.headers == {'Content-Length': '19', 'Content-Type': 'application/json'}
    assert request.content == '{"Hello": "world!"}'

def test_urlencoded_content():
    if False:
        print('Hello World!')
    request = HttpRequest('POST', url='http://example.org', data={'Hello': 'world!'})
    assert request.headers == {'Content-Type': 'application/x-www-form-urlencoded'}

@pytest.mark.parametrize('key', (1, 2.3, None))
def test_multipart_invalid_key(key):
    if False:
        for i in range(10):
            print('nop')
    data = {key: 'abc'}
    files = {'file': io.BytesIO(b'<file content>')}
    with pytest.raises(TypeError) as e:
        HttpRequest(url='http://localhost:8000/', method='POST', data=data, files=files)
    assert 'Invalid type for data name' in str(e.value)
    assert repr(key) in str(e.value)

def test_multipart_invalid_key_binary_string():
    if False:
        print('Hello World!')
    data = {b'abc': 'abc'}
    files = {'file': io.BytesIO(b'<file content>')}
    with pytest.raises(TypeError) as e:
        HttpRequest(url='http://localhost:8000/', method='POST', data=data, files=files)
    assert 'Invalid type for data name' in str(e.value)
    assert repr(b'abc') in str(e.value)

def test_data_str_input():
    if False:
        while True:
            i = 10
    data = {'scope': 'fake_scope', 'grant_type': 'refresh_token', 'refresh_token': 'REDACTED', 'service': 'fake_url.azurecr.io'}
    request = HttpRequest('POST', 'http://localhost:3000/', data=data)
    assert len(request.content) == 4
    assert request.content['scope'] == 'fake_scope'
    assert request.content['grant_type'] == 'refresh_token'
    assert request.content['refresh_token'] == 'REDACTED'
    assert request.content['service'] == 'fake_url.azurecr.io'
    assert len(request.headers) == 1
    assert request.headers['Content-Type'] == 'application/x-www-form-urlencoded'

def test_content_str_input():
    if False:
        return 10
    requests = [HttpRequest('POST', '/fake', content='hello, world!'), HttpRequest('POST', '/fake', content='hello, world!')]
    for request in requests:
        assert len(request.headers) == 2
        assert request.headers['Content-Type'] == 'text/plain'
        assert request.headers['Content-Length'] == '13'
        assert request.content == 'hello, world!'

@pytest.mark.parametrize('value', (object(), {'key': 'value'}))
def test_multipart_invalid_value(value):
    if False:
        while True:
            i = 10
    data = {'text': value}
    files = {'file': io.BytesIO(b'<file content>')}
    with pytest.raises(TypeError) as e:
        HttpRequest('POST', 'http://localhost:8000/', data=data, files=files)
    assert 'Invalid type for data value' in str(e.value)

def test_empty_request():
    if False:
        return 10
    request = HttpRequest('POST', url='http://example.org', data={}, files={})
    assert request.headers == {}
    assert not request.content

def test_read_content(assert_iterator_body):
    if False:
        print('Hello World!')

    def content():
        if False:
            print('Hello World!')
        yield b'test 123'
    request = HttpRequest('POST', 'http://example.org', content=content())
    assert_iterator_body(request, b'test 123')
    assert isinstance(request._data, collections.Iterable)

def test_complicated_json(client):
    if False:
        while True:
            i = 10
    input = {'EmptyByte': '', 'EmptyUnicode': '', 'SpacesOnlyByte': '   ', 'SpacesOnlyUnicode': '   ', 'SpacesBeforeByte': '   Text', 'SpacesBeforeUnicode': '   Text', 'SpacesAfterByte': 'Text   ', 'SpacesAfterUnicode': 'Text   ', 'SpacesBeforeAndAfterByte': '   Text   ', 'SpacesBeforeAndAfterUnicode': '   Text   ', '啊齄丂狛': 'ꀕ', 'RowKey': 'test2', '啊齄丂狛狜': 'hello', 'singlequote': "a''''b", 'doublequote': 'a""""b', 'None': None}
    request = HttpRequest('POST', '/basic/complicated-json', json=input)
    r = client.send_request(request)
    r.raise_for_status()

def test_use_custom_json_encoder():
    if False:
        return 10
    request = HttpRequest('GET', '/headers', json=bytearray('mybytes', 'utf-8'))
    assert request.content == '"bXlieXRlcw=="'

def test_request_policies_raw_request_hook(port):
    if False:
        i = 10
        return i + 15
    request = HttpRequest('GET', '/headers')

    def callback(request):
        if False:
            while True:
                i = 10
        assert is_rest(request.http_request)
        raise ValueError('I entered the callback!')
    custom_hook_policy = CustomHookPolicy(raw_request_hook=callback)
    policies = [UserAgentPolicy('myuseragent'), custom_hook_policy]
    client = MockRestClient(port=port, policies=policies)
    with pytest.raises(ValueError) as ex:
        client.send_request(request)
    assert 'I entered the callback!' in str(ex.value)

def test_request_policies_chain(port):
    if False:
        print('Hello World!')

    class OldPolicyModifyBody(SansIOHTTPPolicy):

        def on_request(self, request):
            if False:
                print('Hello World!')
            assert is_rest(request.http_request)
            request.http_request.set_json_body({'hello': 'world'})

    class NewPolicyModifyHeaders(SansIOHTTPPolicy):

        def on_request(self, request):
            if False:
                print('Hello World!')
            assert is_rest(request.http_request)
            assert request.http_request.content == '{"hello": "world"}'
            request.http_request.headers = {'x-ms-date': 'Thu, 14 Jun 2018 16:46:54 GMT', 'Authorization': 'SharedKey account:G4jjBXA7LI/RnWKIOQ8i9xH4p76pAQ+4Fs4R1VxasaE=', 'Content-Length': '0'}

    class OldPolicySerializeRequest(SansIOHTTPPolicy):

        def on_request(self, request):
            if False:
                while True:
                    i = 10
            assert is_rest(request.http_request)
            request.http_request.data = None
            expected = b'DELETE http://localhost:5000/container0/blob0 HTTP/1.1\r\nx-ms-date: Thu, 14 Jun 2018 16:46:54 GMT\r\nAuthorization: SharedKey account:G4jjBXA7LI/RnWKIOQ8i9xH4p76pAQ+4Fs4R1VxasaE=\r\nContent-Length: 0\r\n\r\n'
            assert request.http_request.serialize() == expected
            raise ValueError('Passed through the policies!')
    policies = [OldPolicyModifyBody(), NewPolicyModifyHeaders(), OldPolicySerializeRequest()]
    request = HttpRequest('DELETE', '/container0/blob0')
    client = MockRestClient(port='5000', policies=policies)
    with pytest.raises(ValueError) as ex:
        client.send_request(request, content='I should be overridden')
    assert 'Passed through the policies!' in str(ex.value)

def test_per_call_policies_old_then_new(port):
    if False:
        for i in range(10):
            print('nop')
    config = Configuration()
    retry_policy = RetryPolicy()
    config.retry_policy = retry_policy

    class OldPolicy(SansIOHTTPPolicy):
        """A policy that deals with a rest request thinking that it's an old request"""

        def on_request(self, pipeline_request):
            if False:
                while True:
                    i = 10
            request = pipeline_request.http_request
            assert is_rest(request)
            assert request.body == '{"hello": "world"}'
            request.set_text_body('change to me!')
            return pipeline_request

    class NewPolicy(SansIOHTTPPolicy):

        def on_request(self, pipeline_request):
            if False:
                i = 10
                return i + 15
            request = pipeline_request.http_request
            assert is_rest(request)
            assert request.content == 'change to me!'
            raise ValueError('I entered the policies!')
    pipeline_client = PipelineClient(base_url='http://localhost:{}'.format(port), config=config, per_call_policies=[OldPolicy(), NewPolicy()])
    client = MockRestClient(port=port)
    client._client = pipeline_client
    with pytest.raises(ValueError) as ex:
        client.send_request(HttpRequest('POST', '/basic/anything', json={'hello': 'world'}))
    assert 'I entered the policies!' in str(ex.value)

def test_json_file_valid():
    if False:
        return 10
    json_bytes = bytearray('{"more": "cowbell"}', encoding='utf-8')
    with io.BytesIO(json_bytes) as json_file:
        request = HttpRequest('PUT', '/fake', json=json_file)
        assert request.headers == {'Content-Type': 'application/json'}
        assert request.content == json_file
        assert not request.content.closed
        assert request.content.read() == b'{"more": "cowbell"}'

def test_json_file_invalid():
    if False:
        while True:
            i = 10
    json_bytes = bytearray('{"more": "cowbell" i am not valid', encoding='utf-8')
    with io.BytesIO(json_bytes) as json_file:
        request = HttpRequest('PUT', '/fake', json=json_file)
        assert request.headers == {'Content-Type': 'application/json'}
        assert request.content == json_file
        assert not request.content.closed
        assert request.content.read() == b'{"more": "cowbell" i am not valid'

def test_json_file_content_type_input():
    if False:
        for i in range(10):
            print('nop')
    json_bytes = bytearray('{"more": "cowbell"}', encoding='utf-8')
    with io.BytesIO(json_bytes) as json_file:
        request = HttpRequest('PUT', '/fake', json=json_file, headers={'Content-Type': 'application/json-special'})
        assert request.headers == {'Content-Type': 'application/json-special'}
        assert request.content == json_file
        assert not request.content.closed
        assert request.content.read() == b'{"more": "cowbell"}'

class NonSeekableStream:

    def __init__(self, wrapped_stream):
        if False:
            for i in range(10):
                print('nop')
        self.wrapped_stream = wrapped_stream

    def write(self, data):
        if False:
            while True:
                i = 10
        self.wrapped_stream.write(data)

    def read(self, count):
        if False:
            i = 10
            return i + 15
        raise ValueError('Request should not read me!')

    def seek(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError("Can't seek!")

    def tell(self):
        if False:
            for i in range(10):
                print('nop')
        return self.wrapped_stream.tell()

def test_non_seekable_stream_input():
    if False:
        print('Hello World!')
    data = b'a' * 4 * 1024
    data_stream = NonSeekableStream(io.BytesIO(data))
    HttpRequest(method='PUT', url='http://www.example.com', content=data_stream)

class Stream:

    def __init__(self, length, initial_buffer_length=4 * 1024):
        if False:
            while True:
                i = 10
        self._base_data = os.urandom(initial_buffer_length)
        self._base_data_length = initial_buffer_length
        self._position = 0
        self._remaining = length

    def read(self, size=None):
        if False:
            print('Hello World!')
        raise ValueError('Request should not read me!')

    def remaining(self):
        if False:
            print('Hello World!')
        return self._remaining

def test_stream_input():
    if False:
        i = 10
        return i + 15
    data_stream = Stream(length=4)
    HttpRequest(method='PUT', url='http://www.example.com', content=data_stream)