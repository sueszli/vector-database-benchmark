import logging
import os
from flask.testing import FlaskClient
import pytest
import app
logger = logging.getLogger()

def setup_test_env():
    if False:
        i = 10
        return i + 15
    os.environ['DB_USER'] = os.environ['POSTGRES_USER']
    os.environ['DB_PASS'] = os.environ['POSTGRES_PASSWORD']
    os.environ['DB_NAME'] = os.environ['POSTGRES_DATABASE']
    os.environ['DB_PORT'] = os.environ['POSTGRES_PORT']
    os.environ['INSTANCE_UNIX_SOCKET'] = os.environ['POSTGRES_UNIX_SOCKET']
    os.environ['INSTANCE_HOST'] = os.environ['POSTGRES_INSTANCE_HOST']
    os.environ['INSTANCE_CONNECTION_NAME'] = os.environ['POSTGRES_INSTANCE']

@pytest.fixture(scope='module')
def client() -> FlaskClient:
    if False:
        return 10
    setup_test_env()
    app.app.testing = True
    client = app.app.test_client()
    return client

def test_get_votes(client: FlaskClient) -> None:
    if False:
        while True:
            i = 10
    response = client.get('/')
    text = 'Tabs VS Spaces'
    body = response.text
    assert response.status_code == 200
    assert text in body

def test_cast_vote(client: FlaskClient) -> None:
    if False:
        while True:
            i = 10
    response = client.post('/votes', data={'team': 'SPACES'})
    text = "Vote successfully cast for 'SPACES'"
    body = response.text
    assert response.status_code == 200
    assert text in body

def test_unix_connection(client: FlaskClient) -> None:
    if False:
        i = 10
        return i + 15
    del os.environ['INSTANCE_HOST']
    app.db = app.init_connection_pool()
    assert 'unix_sock' in str(app.db.url)
    test_get_votes(client)
    test_cast_vote(client)

def test_connector_connection(client: FlaskClient) -> None:
    if False:
        i = 10
        return i + 15
    del os.environ['INSTANCE_UNIX_SOCKET']
    app.db = app.init_connection_pool()
    assert str(app.db.url) == 'postgresql+pg8000://'
    test_get_votes(client)
    test_cast_vote(client)