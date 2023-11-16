from .fixtures import FILE_CONTENT, FILE_PATH_ARG
from .utils import http

def test_offline():
    if False:
        while True:
            i = 10
    r = http('--offline', 'https://this-should.never-resolve/foo')
    assert 'GET /foo' in r

def test_offline_raw():
    if False:
        print('Hello World!')
    r = http('--offline', '--raw', 'foo bar', 'https://this-should.never-resolve/foo')
    assert 'POST /foo' in r
    assert 'foo bar' in r

def test_offline_raw_empty_should_use_POST():
    if False:
        while True:
            i = 10
    r = http('--offline', '--raw', '', 'https://this-should.never-resolve/foo')
    assert 'POST /foo' in r

def test_offline_form():
    if False:
        while True:
            i = 10
    r = http('--offline', '--form', 'https://this-should.never-resolve/foo', 'foo=bar')
    assert 'POST /foo' in r
    assert 'foo=bar' in r

def test_offline_json():
    if False:
        print('Hello World!')
    r = http('--offline', 'https://this-should.never-resolve/foo', 'foo=bar')
    assert 'POST /foo' in r
    assert r.json == {'foo': 'bar'}

def test_offline_multipart():
    if False:
        for i in range(10):
            print('nop')
    r = http('--offline', '--multipart', 'https://this-should.never-resolve/foo', 'foo=bar')
    assert 'POST /foo' in r
    assert 'name="foo"' in r

def test_offline_from_file():
    if False:
        for i in range(10):
            print('nop')
    r = http('--offline', 'https://this-should.never-resolve/foo', f'@{FILE_PATH_ARG}')
    assert 'POST /foo' in r
    assert FILE_CONTENT in r

def test_offline_chunked():
    if False:
        while True:
            i = 10
    r = http('--offline', '--chunked', '--form', 'https://this-should.never-resolve/foo', 'hello=world')
    assert 'POST /foo' in r
    assert 'Transfer-Encoding: chunked' in r, r
    assert 'hello=world' in r

def test_offline_download():
    if False:
        print('Hello World!')
    'Absence of response should be handled gracefully with --download'
    r = http('--offline', '--download', 'https://this-should.never-resolve/foo')
    assert 'GET /foo' in r