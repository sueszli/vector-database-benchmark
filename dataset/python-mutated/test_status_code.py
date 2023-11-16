import pytest
from integration_tests.helpers.http_methods_helpers import get

@pytest.mark.benchmark
def test_404_status_code(session):
    if False:
        return 10
    get('/404', expected_status_code=404)

@pytest.mark.benchmark
def test_404_not_found(session):
    if False:
        while True:
            i = 10
    r = get('/real/404', expected_status_code=404)
    assert r.text == 'Not found'

@pytest.mark.benchmark
def test_202_status_code(session):
    if False:
        return 10
    get('/202', expected_status_code=202)

@pytest.mark.benchmark
@pytest.mark.parametrize('function_type', ['sync', 'async'])
def test_sync_500_internal_server_error(function_type: str, session):
    if False:
        for i in range(10):
            print('nop')
    get(f'/{function_type}/raise', expected_status_code=500)