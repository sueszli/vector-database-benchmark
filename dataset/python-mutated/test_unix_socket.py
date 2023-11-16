import logging
import os
import sys
from asyncio import AbstractEventLoop, sleep
from string import ascii_lowercase
import httpcore
import httpx
import pytest
from pytest import LogCaptureFixture
from sanic import Sanic
from sanic.compat import use_context
from sanic.request import Request
from sanic.response import text
pytestmark = pytest.mark.skipif(os.name != 'posix', reason='UNIX only')
SOCKPATH = '/tmp/sanictest.sock'
SOCKPATH2 = '/tmp/sanictest2.sock'
httpx_version = tuple(map(int, httpx.__version__.strip(ascii_lowercase).split('.')))

@pytest.fixture(autouse=True)
def socket_cleanup():
    if False:
        print('Hello World!')
    try:
        os.unlink(SOCKPATH)
    except FileNotFoundError:
        pass
    try:
        os.unlink(SOCKPATH2)
    except FileNotFoundError:
        pass
    yield
    try:
        os.unlink(SOCKPATH2)
    except FileNotFoundError:
        pass
    try:
        os.unlink(SOCKPATH)
    except FileNotFoundError:
        pass

@pytest.mark.xfail(reason='Flaky Test on Non Linux Infra')
def test_unix_socket_creation(caplog: LogCaptureFixture):
    if False:
        for i in range(10):
            print('nop')
    from socket import AF_UNIX, socket
    with socket(AF_UNIX) as sock:
        sock.bind(SOCKPATH)
    assert os.path.exists(SOCKPATH)
    ino = os.stat(SOCKPATH).st_ino
    app = Sanic(name='test')

    @app.after_server_start
    def running(app: Sanic):
        if False:
            i = 10
            return i + 15
        assert os.path.exists(SOCKPATH)
        assert ino != os.stat(SOCKPATH).st_ino
        app.stop()
    with caplog.at_level(logging.INFO):
        app.run(unix=SOCKPATH, single_process=True)
    assert ('sanic.root', logging.INFO, f"Goin' Fast @ {SOCKPATH} http://...") in caplog.record_tuples
    assert not os.path.exists(SOCKPATH)

@pytest.mark.parametrize('path', ('.', 'no-such-directory/sanictest.sock'))
def test_invalid_paths(path: str):
    if False:
        return 10
    app = Sanic(name='test')
    with pytest.raises((FileExistsError, FileNotFoundError)):
        app.run(unix=path, single_process=True)

def test_dont_replace_file():
    if False:
        print('Hello World!')
    with open(SOCKPATH, 'w') as f:
        f.write('File, not socket')
    app = Sanic(name='test')

    @app.after_server_start
    def stop(app: Sanic):
        if False:
            print('Hello World!')
        app.stop()
    with pytest.raises(FileExistsError):
        app.run(unix=SOCKPATH, single_process=True)

def test_dont_follow_symlink():
    if False:
        for i in range(10):
            print('nop')
    from socket import AF_UNIX, socket
    with socket(AF_UNIX) as sock:
        sock.bind(SOCKPATH2)
    os.symlink(SOCKPATH2, SOCKPATH)
    app = Sanic(name='test')

    @app.after_server_start
    def stop(app: Sanic):
        if False:
            i = 10
            return i + 15
        app.stop()
    with pytest.raises(FileExistsError):
        app.run(unix=SOCKPATH, single_process=True)

def test_socket_deleted_while_running():
    if False:
        while True:
            i = 10
    app = Sanic(name='test')

    @app.after_server_start
    async def hack(app: Sanic):
        os.unlink(SOCKPATH)
        app.stop()
    app.run(host='myhost.invalid', unix=SOCKPATH, single_process=True)

def test_socket_replaced_with_file():
    if False:
        for i in range(10):
            print('nop')
    app = Sanic(name='test')

    @app.after_server_start
    async def hack(app: Sanic):
        os.unlink(SOCKPATH)
        with open(SOCKPATH, 'w') as f:
            f.write('Not a socket')
        app.stop()
    app.run(host='myhost.invalid', unix=SOCKPATH, single_process=True)

def test_unix_connection():
    if False:
        while True:
            i = 10
    app = Sanic(name='test')

    @app.get('/')
    def handler(request: Request):
        if False:
            while True:
                i = 10
        return text(f'{request.conn_info.server}')

    @app.after_server_start
    async def client(app: Sanic):
        if httpx_version >= (0, 20):
            transport = httpx.AsyncHTTPTransport(uds=SOCKPATH)
        else:
            transport = httpcore.AsyncConnectionPool(uds=SOCKPATH)
        try:
            async with httpx.AsyncClient(transport=transport) as client:
                r = await client.get('http://myhost.invalid/')
                assert r.status_code == 200
                assert r.text == os.path.abspath(SOCKPATH)
        finally:
            app.stop()
    app.run(host='myhost.invalid', unix=SOCKPATH, single_process=True)

def handler(request: Request):
    if False:
        return 10
    return text(f'{request.conn_info.server}')

async def client(app: Sanic, loop: AbstractEventLoop):
    try:
        transport = httpx.AsyncHTTPTransport(uds=SOCKPATH)
        async with httpx.AsyncClient(transport=transport) as client:
            r = await client.get('http://myhost.invalid/')
            assert r.status_code == 200
            assert r.text == os.path.abspath(SOCKPATH)
    finally:
        await sleep(0.2)
        app.stop()

@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='This test requires fork context')
def test_unix_connection_multiple_workers():
    if False:
        return 10
    with use_context('fork'):
        app_multi = Sanic(name='test')
        app_multi.get('/')(handler)
        app_multi.listener('after_server_start')(client)
        app_multi.run(host='myhost.invalid', unix=SOCKPATH, workers=2)