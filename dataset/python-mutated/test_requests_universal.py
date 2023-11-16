import concurrent.futures
import requests.utils
import pytest
from azure.core.pipeline.transport import RequestsTransport
from utils import HTTP_REQUESTS, REQUESTS_TRANSPORT_RESPONSES, create_transport_response
from azure.core.pipeline._tools import is_rest

def test_threading_basic_requests():
    if False:
        print('Hello World!')
    sender = RequestsTransport()
    main_thread_session = sender.session

    def thread_body(local_sender):
        if False:
            for i in range(10):
                print('nop')
        assert local_sender.session is main_thread_session
        return True
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(thread_body, sender)
        assert future.result()

@pytest.mark.parametrize('http_request', HTTP_REQUESTS)
def test_requests_auto_headers(port, http_request):
    if False:
        print('Hello World!')
    request = http_request('POST', 'http://localhost:{}/basic/string'.format(port))
    with RequestsTransport() as sender:
        response = sender.send(request)
        auto_headers = response.internal_response.request.headers
        assert 'Content-Type' not in auto_headers

def _create_requests_response(http_response, body_bytes, headers=None):
    if False:
        for i in range(10):
            print('nop')
    req_response = requests.Response()
    req_response._content = body_bytes
    req_response._content_consumed = True
    req_response.status_code = 200
    req_response.reason = 'OK'
    if headers:
        req_response.headers.update(headers)
    req_response.encoding = requests.utils.get_encoding_from_headers(req_response.headers)
    response = create_transport_response(http_response, None, req_response)
    return response

@pytest.mark.parametrize('http_response', REQUESTS_TRANSPORT_RESPONSES)
def test_requests_response_text(http_response):
    if False:
        while True:
            i = 10
    for encoding in ['utf-8', 'utf-8-sig', None]:
        res = _create_requests_response(http_response, b'\xef\xbb\xbf56', {'Content-Type': 'text/plain'})
        if is_rest(http_response):
            res.read()
        assert res.text(encoding) == '56', "Encoding {} didn't work".format(encoding)

@pytest.mark.parametrize('http_response', REQUESTS_TRANSPORT_RESPONSES)
def test_repr(http_response):
    if False:
        i = 10
        return i + 15
    res = _create_requests_response(http_response, b'\xef\xbb\xbf56', {'Content-Type': 'text/plain'})
    class_name = 'HttpResponse' if is_rest(http_response) else 'RequestsTransportResponse'
    assert repr(res) == '<{}: 200 OK, Content-Type: text/plain>'.format(class_name)