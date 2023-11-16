import pytest
from integration_tests.helpers.http_methods_helpers import get

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_middlewares(function_type: str, session):
    if False:
        return 10
    r = get(f'/{function_type}/middlewares')
    assert 'before' not in r.headers
    assert 'after' in r.headers
    assert r.headers['after'] == f'{function_type}_after_request'
    assert r.text == f'{function_type} middlewares after'

@pytest.mark.benchmark
def test_global_middleware(session):
    if False:
        for i in range(10):
            print('nop')
    r = get('/sync/global/middlewares')
    assert 'global_before' not in r.headers
    assert 'global_after' in r.headers
    assert r.headers['global_after'] == 'global_after_request'
    assert r.text == 'sync global middlewares'

@pytest.mark.benchmark
def test_response_in_before_middleware(session):
    if False:
        i = 10
        return i + 15
    r = get('/sync/middlewares/401', should_check_response=False)
    assert r.status_code == 401