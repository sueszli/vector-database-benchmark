import http.client
import json
import os
import shutil
import socket
import ssl
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import pytest
from watchdog.events import EVENT_TYPE_MODIFIED
from watchdog.events import EVENT_TYPE_OPENED
from watchdog.events import FileModifiedEvent
from werkzeug import run_simple
from werkzeug._reloader import _find_stat_paths
from werkzeug._reloader import _find_watchdog_paths
from werkzeug._reloader import _get_args_for_reloading
from werkzeug._reloader import WatchdogReloaderLoop
from werkzeug.datastructures import FileStorage
from werkzeug.serving import make_ssl_devcert
from werkzeug.test import stream_encode_multipart

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('kwargs', [pytest.param({}, id='http'), pytest.param({'ssl_context': 'adhoc'}, id='https'), pytest.param({'use_reloader': True}, id='reloader'), pytest.param({'hostname': 'unix'}, id='unix socket', marks=pytest.mark.skipif(not hasattr(socket, 'AF_UNIX'), reason='requires unix socket support'))])
@pytest.mark.dev_server
def test_server(tmp_path, dev_server, kwargs: dict):
    if False:
        return 10
    if kwargs.get('hostname') == 'unix':
        kwargs['hostname'] = f"unix://{tmp_path / 'test.sock'}"
    client = dev_server(**kwargs)
    r = client.request()
    assert r.status == 200
    assert r.json['PATH_INFO'] == '/'

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_untrusted_host(standard_app):
    if False:
        while True:
            i = 10
    r = standard_app.request('http://missing.test:1337/index.html#ignore', headers={'x-base-url': standard_app.url})
    assert r.json['HTTP_HOST'] == 'missing.test:1337'
    assert r.json['PATH_INFO'] == '/index.html'
    (host, _, port) = r.json['HTTP_X_BASE_URL'].rpartition(':')
    assert r.json['SERVER_NAME'] == host.partition('http://')[2]
    assert r.json['SERVER_PORT'] == port

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_double_slash_path(standard_app):
    if False:
        while True:
            i = 10
    r = standard_app.request('//double-slash')
    assert 'double-slash' not in r.json['HTTP_HOST']
    assert r.json['PATH_INFO'] == '/double-slash'

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_500_error(standard_app):
    if False:
        i = 10
        return i + 15
    r = standard_app.request('/crash')
    assert r.status == 500
    assert b'Internal Server Error' in r.data

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_ssl_dev_cert(tmp_path, dev_server):
    if False:
        while True:
            i = 10
    client = dev_server(ssl_context=make_ssl_devcert(tmp_path))
    r = client.request()
    assert r.json['wsgi.url_scheme'] == 'https'

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_ssl_object(dev_server):
    if False:
        while True:
            i = 10
    client = dev_server(ssl_context='custom')
    r = client.request()
    assert r.json['wsgi.url_scheme'] == 'https'

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('reloader_type', ['stat', 'watchdog'])
@pytest.mark.skipif(os.name == 'nt' and 'CI' in os.environ, reason='unreliable on Windows during CI')
@pytest.mark.dev_server
def test_reloader_sys_path(tmp_path, dev_server, reloader_type):
    if False:
        return 10
    'This tests the general behavior of the reloader. It also tests\n    that fixing an import error triggers a reload, not just Python\n    retrying the failed import.\n    '
    real_path = tmp_path / 'real_app.py'
    real_path.write_text('syntax error causes import error')
    client = dev_server('reloader', reloader_type=reloader_type)
    assert client.request().status == 500
    shutil.copyfile(Path(__file__).parent / 'live_apps' / 'standard_app.py', real_path)
    client.wait_for_log(f' * Detected change in {str(real_path)!r}, reloading')
    client.wait_for_reload()
    assert client.request().status == 200

@patch.object(WatchdogReloaderLoop, 'trigger_reload')
def test_watchdog_reloader_ignores_opened(mock_trigger_reload):
    if False:
        while True:
            i = 10
    reloader = WatchdogReloaderLoop()
    modified_event = FileModifiedEvent('')
    modified_event.event_type = EVENT_TYPE_MODIFIED
    reloader.event_handler.on_any_event(modified_event)
    mock_trigger_reload.assert_called_once()
    reloader.trigger_reload.reset_mock()
    opened_event = FileModifiedEvent('')
    opened_event.event_type = EVENT_TYPE_OPENED
    reloader.event_handler.on_any_event(opened_event)
    reloader.trigger_reload.assert_not_called()

@pytest.mark.skipif(sys.version_info >= (3, 10), reason='not needed on >= 3.10')
def test_windows_get_args_for_reloading(monkeypatch, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    argv = [str(tmp_path / 'test.exe'), 'run']
    monkeypatch.setattr('sys.executable', str(tmp_path / 'python.exe'))
    monkeypatch.setattr('sys.argv', argv)
    monkeypatch.setattr('__main__.__package__', None)
    monkeypatch.setattr('os.name', 'nt')
    rv = _get_args_for_reloading()
    assert rv == argv

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('find', [_find_stat_paths, _find_watchdog_paths])
def test_exclude_patterns(find):
    if False:
        while True:
            i = 10
    paths = find(set(), set())
    path_to_exclude = next(iter(paths))
    assert any((p.startswith(path_to_exclude) for p in paths))
    paths = find(set(), {f'{path_to_exclude}*'})
    assert not any((p.startswith(path_to_exclude) for p in paths))

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_wrong_protocol(standard_app):
    if False:
        for i in range(10):
            print('nop')
    "An HTTPS request to an HTTP server doesn't show a traceback.\n    https://github.com/pallets/werkzeug/pull/838\n    "
    conn = http.client.HTTPSConnection(standard_app.addr)
    with pytest.raises(ssl.SSLError):
        conn.request('GET', f'https://{standard_app.addr}')
    assert 'Traceback' not in standard_app.log.read()

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_content_type_and_length(standard_app):
    if False:
        print('Hello World!')
    r = standard_app.request()
    assert 'CONTENT_TYPE' not in r.json
    assert 'CONTENT_LENGTH' not in r.json
    r = standard_app.request(body=b'{}', headers={'content-type': 'application/json'})
    assert r.json['CONTENT_TYPE'] == 'application/json'
    assert r.json['CONTENT_LENGTH'] == '2'

def test_port_is_int():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError, match='port must be an integer'):
        run_simple('127.0.0.1', '5000', None)

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.parametrize('send_length', [False, True])
@pytest.mark.dev_server
def test_chunked_request(monkeypatch, dev_server, send_length):
    if False:
        return 10
    (stream, length, boundary) = stream_encode_multipart({'value': 'this is text', 'file': FileStorage(BytesIO(b'this is a file'), filename='test.txt', content_type='text/plain')})
    client = dev_server('data')
    conn = client.connect(blocksize=128)
    conn.putrequest('POST', '/')
    conn.putheader('Transfer-Encoding', 'chunked')
    conn.putheader('Content-Type', f'multipart/form-data; boundary={boundary}')
    if send_length:
        conn.putheader('Content-Length', 'invalid')
        expect_content_len = 'invalid'
    else:
        expect_content_len = None
    conn.endheaders(stream, encode_chunked=True)
    r = conn.getresponse()
    data = json.load(r)
    r.close()
    assert data['form']['value'] == 'this is text'
    assert data['files']['file'] == 'this is a file'
    environ = data['environ']
    assert environ['HTTP_TRANSFER_ENCODING'] == 'chunked'
    assert environ.get('CONTENT_LENGTH') == expect_content_len
    assert environ['wsgi.input_terminated']

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_multiple_headers_concatenated(standard_app):
    if False:
        i = 10
        return i + 15
    'A header key can be sent multiple times. The server will join all\n    the values with commas.\n\n    https://tools.ietf.org/html/rfc3875#section-4.1.18\n    '
    conn = standard_app.connect()
    conn.putrequest('GET', '/')
    conn.putheader('XYZ', 'a ')
    conn.putheader('X-Ignore-1', 'ignore value')
    conn.putheader('XYZ', ' b')
    conn.putheader('X-Ignore-2', 'ignore value')
    conn.putheader('XYZ', 'c ')
    conn.putheader('X-Ignore-3', 'ignore value')
    conn.putheader('XYZ', 'd')
    conn.endheaders()
    r = conn.getresponse()
    data = json.load(r)
    r.close()
    assert data['HTTP_XYZ'] == 'a ,b,c ,d'

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_multiline_header_folding(standard_app):
    if False:
        i = 10
        return i + 15
    'A header value can be split over multiple lines with a leading\n    tab. The server will remove the newlines and preserve the tabs.\n\n    https://tools.ietf.org/html/rfc2616#section-2.2\n    '
    conn = standard_app.connect()
    conn.putrequest('GET', '/')
    conn.putheader('XYZ', 'first', 'second', 'third')
    conn.endheaders()
    r = conn.getresponse()
    data = json.load(r)
    r.close()
    assert data['HTTP_XYZ'] == 'first\tsecond\tthird'

@pytest.mark.parametrize('endpoint', ['', 'crash'])
@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_streaming_close_response(dev_server, endpoint):
    if False:
        return 10
    'When using HTTP/1.0, chunked encoding is not supported. Fall\n    back to Connection: close, but this allows no reliable way to\n    distinguish between complete and truncated responses.\n    '
    r = dev_server('streaming').request('/' + endpoint)
    assert r.getheader('connection') == 'close'
    assert r.data == ''.join((str(x) + '\n' for x in range(5))).encode()

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_streaming_chunked_response(dev_server):
    if False:
        print('Hello World!')
    'When using HTTP/1.1, use Transfer-Encoding: chunked for streamed\n    responses, since it can distinguish the end of the response without\n    closing the connection.\n\n    https://tools.ietf.org/html/rfc2616#section-3.6.1\n    '
    r = dev_server('streaming', threaded=True).request('/')
    assert r.getheader('transfer-encoding') == 'chunked'
    assert r.data == ''.join((str(x) + '\n' for x in range(5))).encode()

@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
@pytest.mark.dev_server
def test_streaming_chunked_truncation(dev_server):
    if False:
        i = 10
        return i + 15
    'When using HTTP/1.1, chunked encoding allows the client to detect\n    content truncated by a prematurely closed connection.\n    '
    with pytest.raises(http.client.IncompleteRead):
        dev_server('streaming', threaded=True).request('/crash')