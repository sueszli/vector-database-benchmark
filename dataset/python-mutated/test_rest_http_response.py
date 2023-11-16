import io
import sys
import pytest
from azure.core.rest import HttpRequest, HttpResponse
from azure.core.rest._requests_basic import RestRequestsTransportResponse
from azure.core.exceptions import HttpResponseError
import xml.etree.ElementTree as ET
from utils import readonly_checks

@pytest.fixture
def send_request(client):
    if False:
        while True:
            i = 10

    def _send_request(request):
        if False:
            print('Hello World!')
        response = client.send_request(request, stream=False)
        response.raise_for_status()
        return response
    return _send_request

def test_response(send_request, port):
    if False:
        print('Hello World!')
    response = send_request(request=HttpRequest('GET', '/basic/string'))
    assert response.status_code == 200
    assert response.reason == 'OK'
    assert response.text() == 'Hello, world!'
    assert response.request.method == 'GET'
    assert response.request.url == 'http://localhost:{}/basic/string'.format(port)

def test_response_content(send_request):
    if False:
        i = 10
        return i + 15
    response = send_request(request=HttpRequest('GET', '/basic/bytes'))
    assert response.status_code == 200
    assert response.reason == 'OK'
    assert response.text() == 'Hello, world!'

def test_response_text(send_request):
    if False:
        print('Hello World!')
    response = send_request(request=HttpRequest('GET', '/basic/string'))
    assert response.status_code == 200
    assert response.reason == 'OK'
    assert response.text() == 'Hello, world!'
    assert response.headers['Content-Length'] == '13'
    assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
    assert response.content_type == 'text/plain; charset=utf-8'

def test_response_html(send_request):
    if False:
        i = 10
        return i + 15
    response = send_request(request=HttpRequest('GET', '/basic/html'))
    assert response.status_code == 200
    assert response.reason == 'OK'
    assert response.text() == '<html><body>Hello, world!</html></body>'

def test_raise_for_status(client):
    if False:
        for i in range(10):
            print('nop')
    response = client.send_request(HttpRequest('GET', '/basic/string'))
    response.raise_for_status()
    response = client.send_request(HttpRequest('GET', '/errors/403'))
    assert response.status_code == 403
    with pytest.raises(HttpResponseError):
        response.raise_for_status()
    response = client.send_request(HttpRequest('GET', '/errors/500'), retry_total=0)
    assert response.status_code == 500
    with pytest.raises(HttpResponseError):
        response.raise_for_status()

def test_response_repr(send_request):
    if False:
        i = 10
        return i + 15
    response = send_request(request=HttpRequest('GET', '/basic/string'))
    assert repr(response) == '<HttpResponse: 200 OK, Content-Type: text/plain; charset=utf-8>'

def test_response_content_type_encoding(send_request):
    if False:
        print('Hello World!')
    '\n    Use the charset encoding in the Content-Type header if possible.\n    '
    response = send_request(request=HttpRequest('GET', '/encoding/latin-1'))
    assert response.content_type == 'text/plain; charset=latin-1'
    assert response.text() == 'Latin 1: ÿ'
    assert response.encoding == 'latin-1'

def test_response_autodetect_encoding(send_request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Autodetect encoding if there is no Content-Type header.\n    '
    response = send_request(request=HttpRequest('GET', '/encoding/latin-1'))
    assert response.text() == 'Latin 1: ÿ'
    assert response.encoding == 'latin-1'

def test_response_fallback_to_autodetect(send_request):
    if False:
        print('Hello World!')
    '\n    Fallback to autodetection if we get an invalid charset in the Content-Type header.\n    '
    response = send_request(request=HttpRequest('GET', '/encoding/invalid-codec-name'))
    assert response.headers['Content-Type'] == 'text/plain; charset=invalid-codec-name'
    assert response.text() == 'おはようございます。'
    assert response.encoding is None

def test_response_no_charset_with_ascii_content(send_request):
    if False:
        while True:
            i = 10
    '\n    A response with ascii encoded content should decode correctly,\n    even with no charset specified.\n    '
    response = send_request(request=HttpRequest('GET', '/encoding/no-charset'))
    assert response.headers['Content-Type'] == 'text/plain'
    assert response.status_code == 200
    assert response.encoding is None
    assert response.text() == 'Hello, world!'

def test_response_no_charset_with_iso_8859_1_content(send_request):
    if False:
        while True:
            i = 10
    "\n    We don't support iso-8859-1 by default following conversations\n    about encoding flow\n    "
    response = send_request(request=HttpRequest('GET', '/encoding/iso-8859-1'))
    assert response.text() == 'Accented: �sterreich'
    assert response.encoding is None

def test_json(send_request):
    if False:
        while True:
            i = 10
    response = send_request(request=HttpRequest('GET', '/basic/json'))
    assert response.json() == {'greeting': 'hello', 'recipient': 'world'}
    assert response.encoding is None

def test_json_with_specified_encoding(send_request):
    if False:
        for i in range(10):
            print('nop')
    response = send_request(request=HttpRequest('GET', '/encoding/json'))
    assert response.json() == {'greeting': 'hello', 'recipient': 'world'}
    assert response.encoding == 'utf-16'

def test_emoji(send_request):
    if False:
        return 10
    response = send_request(request=HttpRequest('GET', '/encoding/emoji'))
    assert response.text() == '👩'

def test_emoji_family_with_skin_tone_modifier(send_request):
    if False:
        for i in range(10):
            print('nop')
    response = send_request(request=HttpRequest('GET', '/encoding/emoji-family-skin-tone-modifier'))
    assert response.text() == '👩🏻\u200d👩🏽\u200d👧🏾\u200d👦🏿 SSN: 859-98-0987'

def test_korean_nfc(send_request):
    if False:
        while True:
            i = 10
    response = send_request(request=HttpRequest('GET', '/encoding/korean'))
    assert response.text() == '아가'

def test_urlencoded_content(send_request):
    if False:
        for i in range(10):
            print('nop')
    send_request(request=HttpRequest('POST', '/urlencoded/pet/add/1', data={'pet_type': 'dog', 'pet_food': 'meat', 'name': 'Fido', 'pet_age': 42}))

def test_multipart_files_content(send_request):
    if False:
        for i in range(10):
            print('nop')
    request = HttpRequest('POST', '/multipart/basic', files={'fileContent': io.BytesIO(b'<file content>')})
    send_request(request)

def test_multipart_data_and_files_content(send_request):
    if False:
        for i in range(10):
            print('nop')
    request = HttpRequest('POST', '/multipart/data-and-files', data={'message': 'Hello, world!'}, files={'fileContent': io.BytesIO(b'<file content>')})
    send_request(request)

def test_multipart_encode_non_seekable_filelike(send_request):
    if False:
        print('Hello World!')
    '\n    Test that special readable but non-seekable filelike objects are supported,\n    at the cost of reading them into memory at most once.\n    '

    class IteratorIO(io.IOBase):

        def __init__(self, iterator):
            if False:
                i = 10
                return i + 15
            self._iterator = iterator

        def read(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            return b''.join(self._iterator)

    def data():
        if False:
            print('Hello World!')
        yield b'Hello'
        yield b'World'
    fileobj = IteratorIO(data())
    files = {'file': fileobj}
    request = HttpRequest('POST', '/multipart/non-seekable-filelike', files=files)
    send_request(request)

def test_get_xml_basic(send_request):
    if False:
        while True:
            i = 10
    request = HttpRequest('GET', '/xml/basic')
    response = send_request(request)
    parsed_xml = ET.fromstring(response.text())
    assert parsed_xml.tag == 'slideshow'
    attributes = parsed_xml.attrib
    assert attributes['title'] == 'Sample Slide Show'
    assert attributes['date'] == 'Date of publication'
    assert attributes['author'] == 'Yours Truly'

def test_put_xml_basic(send_request):
    if False:
        i = 10
        return i + 15
    basic_body = '<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n<slideshow\n        title="Sample Slide Show"\n        date="Date of publication"\n        author="Yours Truly">\n    <slide type="all">\n        <title>Wake up to WonderWidgets!</title>\n    </slide>\n    <slide type="all">\n        <title>Overview</title>\n        <item>Why WonderWidgets are great</item>\n        <item></item>\n        <item>Who buys WonderWidgets</item>\n    </slide>\n</slideshow>'
    request = HttpRequest('PUT', '/xml/basic', content=ET.fromstring(basic_body))
    send_request(request)

def test_send_request_return_pipeline_response(client):
    if False:
        for i in range(10):
            print('nop')
    request = HttpRequest('GET', '/basic/string')
    response = client.send_request(request, _return_pipeline_response=True)
    assert hasattr(response, 'http_request')
    assert hasattr(response, 'http_response')
    assert hasattr(response, 'context')
    assert response.http_response.text() == 'Hello, world!'
    assert hasattr(response.http_request, 'content')

def test_text_and_encoding(send_request):
    if False:
        i = 10
        return i + 15
    response = send_request(request=HttpRequest('GET', '/encoding/emoji'))
    assert response.content == '👩'.encode('utf-8')
    assert response.text() == '👩'
    response.encoding = 'utf-16'
    assert response.text() == '鿰ꦑ' == response.content.decode(response.encoding)
    assert response.text('latin-1') == 'ð\x9f\x91©' == response.content.decode('latin-1')
    assert response.encoding == 'utf-16'

def test_passing_encoding_to_text(send_request):
    if False:
        while True:
            i = 10
    response = send_request(request=HttpRequest('GET', '/encoding/emoji'))
    assert response.content == '👩'.encode('utf-8')
    assert response.text() == '👩'
    assert response.text('latin-1') == 'ð\x9f\x91©'
    assert response.text() == '👩'

def test_initialize_response_abc():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError) as ex:
        HttpResponse()
    assert "Can't instantiate abstract class" in str(ex)

def test_readonly(send_request):
    if False:
        return 10
    'Make sure everything that is readonly is readonly'
    response = send_request(HttpRequest('GET', '/health'))
    assert isinstance(response, RestRequestsTransportResponse)
    from azure.core.pipeline.transport import RequestsTransportResponse
    readonly_checks(response, old_response_class=RequestsTransportResponse)