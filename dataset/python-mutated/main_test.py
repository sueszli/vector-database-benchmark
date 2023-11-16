import pytest
import main

@pytest.fixture
def client():
    if False:
        i = 10
        return i + 15
    main.app.testing = True
    return main.app.test_client()

def test_empty_query_string(client):
    if False:
        print('Hello World!')
    r = client.get('/diagram.png')
    assert r.status_code == 400

def test_empty_dot_parameter(client):
    if False:
        print('Hello World!')
    r = client.get('/diagram.png?dot=')
    assert r.status_code == 400

def test_bad_dot_parameter(client):
    if False:
        return 10
    r = client.get('/diagram.png?dot=digraph')
    assert r.status_code == 400

def test_good_dot_parameter(client):
    if False:
        return 10
    r = client.get('/diagram.png?dot=digraph G { A -> {B, C, D} -> {F} }')
    assert r.content_type == 'image/png'