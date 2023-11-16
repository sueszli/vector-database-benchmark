import pytest
from azure.core.rest import HttpRequest
import collections.abc

@pytest.fixture
def assert_aiterator_body():
    if False:
        i = 10
        return i + 15

    async def _comparer(request, final_value):
        parts = []
        async for part in request.content:
            parts.append(part)
        content = b''.join(parts)
        assert content == final_value
    return _comparer

def test_transfer_encoding_header():
    if False:
        return 10

    async def streaming_body(data):
        yield data
    data = streaming_body(b'test 123')
    request = HttpRequest('POST', 'http://example.org', data=data)
    assert 'Content-Length' not in request.headers

def test_override_content_length_header():
    if False:
        print('Hello World!')

    async def streaming_body(data):
        yield data
    data = streaming_body(b'test 123')
    headers = {'Content-Length': '0'}
    request = HttpRequest('POST', 'http://example.org', data=data, headers=headers)
    assert request.headers['Content-Length'] == '0'

@pytest.mark.asyncio
async def test_aiterable_content(assert_aiterator_body):

    class Content:

        async def __aiter__(self):
            yield b'test 123'
    request = HttpRequest('POST', 'http://example.org', content=Content())
    assert request.headers == {}
    await assert_aiterator_body(request, b'test 123')

@pytest.mark.asyncio
async def test_aiterator_content(assert_aiterator_body):

    async def hello_world():
        yield b'Hello, '
        yield b'world!'
    request = HttpRequest('POST', url='http://example.org', content=hello_world())
    assert not isinstance(request._data, collections.abc.Iterable)
    assert isinstance(request._data, collections.abc.AsyncIterable)
    assert request.headers == {}
    await assert_aiterator_body(request, b'Hello, world!')
    request = HttpRequest('POST', url='http://example.org', data=hello_world())
    assert not isinstance(request._data, collections.abc.Iterable)
    assert isinstance(request._data, collections.abc.AsyncIterable)
    assert request.headers == {}
    await assert_aiterator_body(request, b'Hello, world!')
    request = HttpRequest('GET', url='http://example.org', data=hello_world())
    assert not isinstance(request._data, collections.abc.Iterable)
    assert isinstance(request._data, collections.abc.AsyncIterable)
    assert request.headers == {}
    await assert_aiterator_body(request, b'Hello, world!')

@pytest.mark.asyncio
async def test_read_content(assert_aiterator_body):

    async def content():
        yield b'test 123'
    request = HttpRequest('POST', 'http://example.org', content=content())
    await assert_aiterator_body(request, b'test 123')
    assert isinstance(request._data, collections.abc.AsyncIterable)