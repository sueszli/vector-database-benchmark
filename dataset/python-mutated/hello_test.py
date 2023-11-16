import pytest
import hello

@pytest.fixture
def client():
    if False:
        print('Hello World!')
    hello.app.testing = True
    return hello.app.test_client()

def test_home_page(client):
    if False:
        for i in range(10):
            print('nop')
    response = client.get('/')
    assert response.status_code == 200
    assert response.text.startswith('Hello. This page was last updated at ')
    assert response.text.endswith('2:19 PM PST, Monday, November 6, 2023.')

def test_other_page(client):
    if False:
        while True:
            i = 10
    response = client.get('/help')
    assert response.status_code == 404