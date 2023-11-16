from __future__ import annotations
import pytest
import werkzeug.test
import werkzeug.wrappers
from airflow.www.app import create_app
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def app():
    if False:
        for i in range(10):
            print('nop')

    @conf_vars({('webserver', 'base_url'): 'http://localhost/test'})
    def factory():
        if False:
            for i in range(10):
                print('nop')
        return create_app(testing=True)
    app = factory()
    app.config['WTF_CSRF_ENABLED'] = False
    return app

@pytest.fixture()
def client(app):
    if False:
        i = 10
        return i + 15
    return werkzeug.test.Client(app, werkzeug.wrappers.response.Response)

def test_mount(client):
    if False:
        return 10
    resp = client.get('/test/health')
    assert resp.status_code == 200
    assert b'healthy' in resp.data

def test_not_found(client):
    if False:
        print('Hello World!')
    resp = client.get('/', follow_redirects=True)
    assert resp.status_code == 404

def test_index(client):
    if False:
        i = 10
        return i + 15
    resp = client.get('/test/')
    assert resp.status_code == 302
    assert resp.headers['Location'] == '/test/home'