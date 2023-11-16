import pytest
from requests import Response
from integration_tests.helpers.http_methods_helpers import get

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_param(function_type: str, session):
    if False:
        for i in range(10):
            print('nop')
    r = get(f'/{function_type}/param/1')
    assert r.text == '1'
    r = get(f'/{function_type}/param/12345')
    assert r.text == '12345'

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_param_suffix(function_type: str, session):
    if False:
        print('Hello World!')
    r = get(f'/{function_type}/extra/foo/1/baz')
    assert r.text == 'foo/1/baz'
    r = get(f'/{function_type}/extra/foo/bar/baz')
    assert r.text == 'foo/bar/baz'

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_serve_html(function_type: str, session):
    if False:
        i = 10
        return i + 15

    def check_response(r: Response):
        if False:
            while True:
                i = 10
        assert r.text.startswith('<!DOCTYPE html>')
        assert 'Hello world. How are you?' in r.text
    check_response(get(f'/{function_type}/serve/html'))

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_template(function_type: str, session):
    if False:
        print('Hello World!')

    def check_response(r: Response):
        if False:
            for i in range(10):
                print('nop')
        assert r.text.startswith('\n\n<!DOCTYPE html>')
        assert 'Jinja2' in r.text
        assert 'Robyn' in r.text
    check_response(get(f'/{function_type}/template'))

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_queries(function_type: str, session):
    if False:
        for i in range(10):
            print('nop')
    r = get(f'/{function_type}/queries?hello=robyn')
    assert r.json() == {'hello': 'robyn'}
    r = get(f'/{function_type}/queries')
    assert r.json() == {}