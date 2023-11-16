import asyncio
from contextlib import contextmanager
import hashlib
import os
import platform
import random
import subprocess
import sys
import time
import httpx
import pytest
import requests
import requests.exceptions
import websockets
import websockets.exceptions
from falcon import testing
from . import _asgi_test_app
_MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
_PYPY = platform.python_implementation() == 'PyPy'
_WIN32 = sys.platform.startswith('win')
_SERVER_HOST = '127.0.0.1'
_SIZE_1_KB = 1024
_SIZE_1_MB = _SIZE_1_KB ** 2
_REQUEST_TIMEOUT = 10

class TestASGIServer:

    def test_get(self, server_base_url):
        if False:
            return 10
        resp = requests.get(server_base_url, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.text == '127.0.0.1'

    def test_put(self, server_base_url):
        if False:
            return 10
        body = '{}'
        resp = requests.put(server_base_url, data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.text == '{}'

    def test_head_405(self, server_base_url):
        if False:
            i = 10
            return i + 15
        body = '{}'
        resp = requests.head(server_base_url, data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 405

    def test_post_multipart_form(self, server_base_url):
        if False:
            for i in range(10):
                print('nop')
        size = random.randint(16 * _SIZE_1_MB, 32 * _SIZE_1_MB)
        data = os.urandom(size)
        digest = hashlib.sha1(data).hexdigest()
        files = {'random': ('random.dat', data), 'message': ('hello.txt', b'Hello, World!\n')}
        resp = requests.post(server_base_url + 'forms', files=files, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.json() == {'message': {'filename': 'hello.txt', 'sha1': '60fde9c2310b0d4cad4dab8d126b04387efba289'}, 'random': {'filename': 'random.dat', 'sha1': digest}}

    def test_post_multiple(self, server_base_url):
        if False:
            for i in range(10):
                print('nop')
        body = testing.rand_string(_SIZE_1_KB // 2, _SIZE_1_KB)
        resp = requests.post(server_base_url, data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.text == body
        assert resp.headers['X-Counter'] == '0'
        time.sleep(1)
        resp = requests.post(server_base_url, data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.headers['X-Counter'] == '2002'

    def test_post_invalid_content_length(self, server_base_url):
        if False:
            i = 10
            return i + 15
        headers = {'Content-Length': 'invalid'}
        try:
            resp = requests.post(server_base_url, headers=headers, timeout=_REQUEST_TIMEOUT)
            assert resp.status_code == 400
        except requests.ConnectionError:
            pass

    def test_post_read_bounded_stream(self, server_base_url):
        if False:
            for i in range(10):
                print('nop')
        body = testing.rand_string(_SIZE_1_KB // 2, _SIZE_1_KB)
        resp = requests.post(server_base_url + 'bucket', data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.text == body

    def test_post_read_bounded_stream_large(self, server_base_url):
        if False:
            print('Hello World!')
        'Test that we can correctly read large bodies chunked server-side.\n\n        ASGI servers typically employ some type of flow control to stream\n        large request bodies to the app. This occurs regardless of whether\n        "chunked" Transfer-Encoding is employed by the client.\n        '
        size_mb = 5
        body = os.urandom(_SIZE_1_MB * size_mb)
        resp = requests.put(server_base_url + 'bucket/drops', data=body, timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        assert resp.json().get('drops') > size_mb
        assert resp.json().get('sha1') == hashlib.sha1(body).hexdigest()

    def test_post_read_bounded_stream_no_body(self, server_base_url):
        if False:
            print('Hello World!')
        resp = requests.post(server_base_url + 'bucket', timeout=_REQUEST_TIMEOUT)
        assert not resp.text

    def test_sse(self, server_base_url):
        if False:
            print('Hello World!')
        resp = requests.get(server_base_url + 'events', timeout=_REQUEST_TIMEOUT)
        assert resp.status_code == 200
        events = resp.text.split('\n\n')
        assert len(events) > 2
        for e in events[:-1]:
            assert e == 'data: hello world'
        assert not events[-1]

    def test_sse_client_disconnects_early(self, server_base_url):
        if False:
            while True:
                i = 10
        'Test that when the client connection is lost, the server task does not hang.\n\n        In the case of SSE, Falcon should detect when the client connection is\n        lost and immediately bail out. Currently this is observable by watching\n        the output of the uvicorn and daphne server processes. Also, the\n        _run_server_isolated() method will fail the test if the server process\n        takes too long to shut down.\n        '
        with pytest.raises(requests.exceptions.ConnectionError):
            requests.get(server_base_url + 'events', timeout=_asgi_test_app.SSE_TEST_MAX_DELAY_SEC / 2)

    @pytest.mark.asyncio
    async def test_stream_chunked_request(self, server_base_url):
        """Regression test for https://github.com/falconry/falcon/issues/2024"""

        async def emitter():
            for _ in range(64):
                yield b'123456789ABCDEF\n'
        async with httpx.AsyncClient() as client:
            resp = await client.put(server_base_url + 'bucket/drops', content=emitter(), timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            assert resp.json().get('drops') >= 1

class TestWebSocket:

    @pytest.mark.asyncio
    @pytest.mark.parametrize('explicit_close', [True, False])
    @pytest.mark.parametrize('close_code', [None, 4321])
    async def test_hello(self, explicit_close, close_code, server_url_events_ws):
        echo_expected = 'Check 1 - 😀'
        extra_headers = {'X-Command': 'recv'}
        if explicit_close:
            extra_headers['X-Close'] = 'True'
        if close_code:
            extra_headers['X-Close-Code'] = str(close_code)
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            got_message = False
            while True:
                try:
                    await ws.send(f'{{"command": "echo", "echo": "{echo_expected}"}}')
                    message_text = await ws.recv()
                    message_echo = await ws.recv()
                    message_binary = await ws.recv()
                except websockets.exceptions.ConnectionClosed as ex:
                    if explicit_close and close_code:
                        assert ex.code == close_code
                    else:
                        assert ex.code == 1000
                    break
                got_message = True
                assert message_text == 'hello world'
                assert message_echo == echo_expected
                assert message_binary == b'hello\x00world'
            assert got_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize('explicit_close', [True, False])
    @pytest.mark.parametrize('close_code', [None, 4040])
    async def test_rejected(self, explicit_close, close_code, server_url_events_ws):
        extra_headers = {'X-Accept': 'reject'}
        if explicit_close:
            extra_headers['X-Close'] = 'True'
        if close_code:
            extra_headers['X-Close-Code'] = str(close_code)
        with pytest.raises(websockets.exceptions.InvalidStatusCode) as exc_info:
            async with websockets.connect(server_url_events_ws, extra_headers=extra_headers):
                pass
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_responder(self, server_url_events_ws):
        server_url_events_ws += '/404'
        with pytest.raises(websockets.exceptions.InvalidStatusCode) as exc_info:
            async with websockets.connect(server_url_events_ws):
                pass
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    @pytest.mark.parametrize('subprotocol, expected', [('*', 'amqp'), ('wamp', 'wamp')])
    async def test_select_subprotocol_known(self, subprotocol, expected, server_url_events_ws):
        extra_headers = {'X-Subprotocol': subprotocol}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers, subprotocols=['amqp', 'wamp']) as ws:
            assert ws.subprotocol == expected

    @pytest.mark.asyncio
    async def test_select_subprotocol_unknown(self, server_url_events_ws):
        extra_headers = {'X-Subprotocol': 'xmpp'}
        try:
            async with websockets.connect(server_url_events_ws, extra_headers=extra_headers, subprotocols=['amqp', 'wamp']):
                pass
            pytest.fail('no error raised')
        except websockets.exceptions.NegotiationError as ex:
            assert 'unsupported subprotocol: xmpp' in str(ex)
        except websockets.exceptions.InvalidMessage:
            pass

    @pytest.mark.asyncio
    async def test_disconnecting_client_early(self, server_url_events_ws):
        ws = await websockets.connect(server_url_events_ws, extra_headers={'X-Close': 'True'})
        await asyncio.sleep(0.2)
        message_text = await ws.recv()
        assert message_text == 'hello world'
        message_binary = await ws.recv()
        assert message_binary == b'hello\x00world'
        await ws.close()
        print('closed')
        await asyncio.sleep(1)

    @pytest.mark.asyncio
    async def test_send_before_accept(self, server_url_events_ws):
        extra_headers = {'x-accept': 'skip'}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            message = await ws.recv()
            assert message == 'OperationNotAllowed'

    @pytest.mark.asyncio
    async def test_recv_before_accept(self, server_url_events_ws):
        extra_headers = {'x-accept': 'skip', 'x-command': 'recv'}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            message = await ws.recv()
            assert message == 'OperationNotAllowed'

    @pytest.mark.asyncio
    async def test_invalid_close_code(self, server_url_events_ws):
        extra_headers = {'x-close': 'True', 'x-close-code': 42}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            start = time.time()
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=1)
                if message == 'ValueError':
                    break
                elapsed = time.time() - start
                assert elapsed < 2

    @pytest.mark.asyncio
    async def test_close_code_on_unhandled_error(self, server_url_events_ws):
        extra_headers = {'x-raise-error': 'generic'}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            await ws.wait_closed()
        assert ws.close_code in {3011, 1011}

    @pytest.mark.asyncio
    async def test_close_code_on_unhandled_http_error(self, server_url_events_ws):
        extra_headers = {'x-raise-error': 'http'}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            await ws.wait_closed()
        assert ws.close_code == 3400

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mismatch', ['send', 'recv'])
    @pytest.mark.parametrize('mismatch_type', ['text', 'data'])
    async def test_type_mismatch(self, mismatch, mismatch_type, server_url_events_ws):
        extra_headers = {'X-Mismatch': mismatch, 'X-Mismatch-Type': mismatch_type}
        async with websockets.connect(server_url_events_ws, extra_headers=extra_headers) as ws:
            if mismatch == 'recv':
                if mismatch_type == 'text':
                    await ws.send(b'hello')
                else:
                    await ws.send('hello')
            await ws.wait_closed()
        assert ws.close_code in {3011, 1011}

    @pytest.mark.asyncio
    async def test_passing_path_params(self, server_base_url_ws):
        expected_feed_id = '1ee7'
        url = f'{server_base_url_ws}feeds/{expected_feed_id}'
        async with websockets.connect(url) as ws:
            feed_id = await ws.recv()
            assert feed_id == expected_feed_id

@contextmanager
def _run_server_isolated(process_factory, host, port):
    if False:
        print('Hello World!')
    print('\n[Starting server process...]')
    server = process_factory(host, port)
    yield server
    if _WIN32:
        import signal
        print('\n[Sending CTRL+C (SIGINT) to server process...]')
        server.send_signal(signal.CTRL_C_EVENT)
        try:
            server.wait(timeout=10)
        except KeyboardInterrupt:
            pass
        except subprocess.TimeoutExpired:
            print('\n[Killing stubborn server process...]')
            server.kill()
            server.communicate()
            pytest.fail('Server process did not exit in a timely manner and had to be killed.')
    else:
        print('\n[Sending SIGTERM to server process...]')
        server.terminate()
        try:
            server.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            print('\n[Killing stubborn server process...]')
            server.kill()
            server.communicate()
            pytest.fail('Server process did not exit in a timely manner and had to be killed.')

def _uvicorn_factory(host, port):
    if False:
        print('Hello World!')
    if _WIN32:
        script = f"\nimport uvicorn\nimport ctypes\nctypes.windll.kernel32.SetConsoleCtrlHandler(None, 0)\nuvicorn.run('_asgi_test_app:application', host='{host}', port={port})\n"
        return subprocess.Popen((sys.executable, '-c', script), cwd=_MODULE_DIR, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    loop_options = ('--http', 'h11', '--loop', 'asyncio') if _PYPY else ()
    options = ('--host', host, '--port', str(port), '--interface', 'asgi3', '_asgi_test_app:application')
    return subprocess.Popen((sys.executable, '-m', 'uvicorn') + loop_options + options, cwd=_MODULE_DIR)

def _daphne_factory(host, port):
    if False:
        print('Hello World!')
    return subprocess.Popen((sys.executable, '-m', 'daphne', '--bind', host, '--port', str(port), '--verbosity', '2', '--access-log', '-', '_asgi_test_app:application'), cwd=_MODULE_DIR)

def _hypercorn_factory(host, port):
    if False:
        while True:
            i = 10
    if _WIN32:
        script = f"\nfrom hypercorn.run import Config, run\nimport ctypes\nctypes.windll.kernel32.SetConsoleCtrlHandler(None, 0)\nconfig = Config()\nconfig.application_path = '_asgi_test_app:application'\nconfig.bind = ['{host}:{port}']\nconfig.accesslog = '-'\nconfig.debug = True\nrun(config)\n"
        return subprocess.Popen((sys.executable, '-c', script), cwd=_MODULE_DIR, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen((sys.executable, '-m', 'hypercorn', '--bind', f'{host}:{port}', '--access-logfile', '-', '--debug', '_asgi_test_app:application'), cwd=_MODULE_DIR)

def _can_run(factory):
    if False:
        return 10
    if _WIN32 and factory == _daphne_factory:
        pytest.skip('daphne does not support windows')
    if factory == _daphne_factory:
        try:
            import daphne
        except Exception:
            pytest.skip('daphne not installed')
    elif factory == _hypercorn_factory:
        try:
            import hypercorn
        except Exception:
            pytest.skip('hypercorn not installed')
    elif factory == _uvicorn_factory:
        try:
            import uvicorn
        except Exception:
            pytest.skip('uvicorn not installed')

@pytest.fixture(params=[_uvicorn_factory, _daphne_factory, _hypercorn_factory])
def server_base_url(request):
    if False:
        i = 10
        return i + 15
    process_factory = request.param
    _can_run(process_factory)
    for i in range(3):
        server_port = testing.get_unused_port()
        base_url = 'http://{}:{}/'.format(_SERVER_HOST, server_port)
        with _run_server_isolated(process_factory, _SERVER_HOST, server_port) as server:
            start_ts = time.time()
            while time.time() - start_ts < 5:
                try:
                    requests.get(base_url, timeout=0.2)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    time.sleep(0.2)
                else:
                    break
            else:
                if server.poll() is None:
                    pytest.fail('Server is not responding to requests')
                else:
                    continue
            yield base_url
        assert server.returncode == 0
        break
    else:
        pytest.fail('Could not start server')

@pytest.fixture
def server_base_url_ws(server_base_url):
    if False:
        while True:
            i = 10
    return server_base_url.replace('http', 'ws')

@pytest.fixture
def server_url_events_ws(server_base_url_ws):
    if False:
        return 10
    return server_base_url_ws + 'events'