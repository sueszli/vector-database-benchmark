from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.testclient import TestClient

class OverrideResponse(JSONResponse):
    media_type = 'application/x-override'
app = FastAPI()
router_a = APIRouter()
router_a_a = APIRouter()
router_a_b_override = APIRouter()
router_b_override = APIRouter()
router_b_a = APIRouter()
router_b_a_c_override = APIRouter()

@app.get('/')
def get_root():
    if False:
        while True:
            i = 10
    return {'msg': 'Hello World'}

@app.get('/override', response_class=PlainTextResponse)
def get_path_override():
    if False:
        print('Hello World!')
    return 'Hello World'

@router_a.get('/')
def get_a():
    if False:
        print('Hello World!')
    return {'msg': 'Hello A'}

@router_a.get('/override', response_class=PlainTextResponse)
def get_a_path_override():
    if False:
        print('Hello World!')
    return 'Hello A'

@router_a_a.get('/')
def get_a_a():
    if False:
        return 10
    return {'msg': 'Hello A A'}

@router_a_a.get('/override', response_class=PlainTextResponse)
def get_a_a_path_override():
    if False:
        return 10
    return 'Hello A A'

@router_a_b_override.get('/')
def get_a_b():
    if False:
        while True:
            i = 10
    return 'Hello A B'

@router_a_b_override.get('/override', response_class=HTMLResponse)
def get_a_b_path_override():
    if False:
        for i in range(10):
            print('nop')
    return 'Hello A B'

@router_b_override.get('/')
def get_b():
    if False:
        i = 10
        return i + 15
    return 'Hello B'

@router_b_override.get('/override', response_class=HTMLResponse)
def get_b_path_override():
    if False:
        while True:
            i = 10
    return 'Hello B'

@router_b_a.get('/')
def get_b_a():
    if False:
        for i in range(10):
            print('nop')
    return 'Hello B A'

@router_b_a.get('/override', response_class=HTMLResponse)
def get_b_a_path_override():
    if False:
        print('Hello World!')
    return 'Hello B A'

@router_b_a_c_override.get('/')
def get_b_a_c():
    if False:
        while True:
            i = 10
    return 'Hello B A C'

@router_b_a_c_override.get('/override', response_class=OverrideResponse)
def get_b_a_c_path_override():
    if False:
        print('Hello World!')
    return {'msg': 'Hello B A C'}
router_b_a.include_router(router_b_a_c_override, prefix='/c', default_response_class=HTMLResponse)
router_b_override.include_router(router_b_a, prefix='/a')
router_a.include_router(router_a_a, prefix='/a')
router_a.include_router(router_a_b_override, prefix='/b', default_response_class=PlainTextResponse)
app.include_router(router_a, prefix='/a')
app.include_router(router_b_override, prefix='/b', default_response_class=PlainTextResponse)
client = TestClient(app)
json_type = 'application/json'
text_type = 'text/plain; charset=utf-8'
html_type = 'text/html; charset=utf-8'
override_type = 'application/x-override'

def test_app():
    if False:
        print('Hello World!')
    with client:
        response = client.get('/')
    assert response.json() == {'msg': 'Hello World'}
    assert response.headers['content-type'] == json_type

def test_app_override():
    if False:
        i = 10
        return i + 15
    with client:
        response = client.get('/override')
    assert response.content == b'Hello World'
    assert response.headers['content-type'] == text_type

def test_router_a():
    if False:
        while True:
            i = 10
    with client:
        response = client.get('/a')
    assert response.json() == {'msg': 'Hello A'}
    assert response.headers['content-type'] == json_type

def test_router_a_override():
    if False:
        while True:
            i = 10
    with client:
        response = client.get('/a/override')
    assert response.content == b'Hello A'
    assert response.headers['content-type'] == text_type

def test_router_a_a():
    if False:
        return 10
    with client:
        response = client.get('/a/a')
    assert response.json() == {'msg': 'Hello A A'}
    assert response.headers['content-type'] == json_type

def test_router_a_a_override():
    if False:
        for i in range(10):
            print('nop')
    with client:
        response = client.get('/a/a/override')
    assert response.content == b'Hello A A'
    assert response.headers['content-type'] == text_type

def test_router_a_b():
    if False:
        print('Hello World!')
    with client:
        response = client.get('/a/b')
    assert response.content == b'Hello A B'
    assert response.headers['content-type'] == text_type

def test_router_a_b_override():
    if False:
        while True:
            i = 10
    with client:
        response = client.get('/a/b/override')
    assert response.content == b'Hello A B'
    assert response.headers['content-type'] == html_type

def test_router_b():
    if False:
        while True:
            i = 10
    with client:
        response = client.get('/b')
    assert response.content == b'Hello B'
    assert response.headers['content-type'] == text_type

def test_router_b_override():
    if False:
        i = 10
        return i + 15
    with client:
        response = client.get('/b/override')
    assert response.content == b'Hello B'
    assert response.headers['content-type'] == html_type

def test_router_b_a():
    if False:
        i = 10
        return i + 15
    with client:
        response = client.get('/b/a')
    assert response.content == b'Hello B A'
    assert response.headers['content-type'] == text_type

def test_router_b_a_override():
    if False:
        for i in range(10):
            print('nop')
    with client:
        response = client.get('/b/a/override')
    assert response.content == b'Hello B A'
    assert response.headers['content-type'] == html_type

def test_router_b_a_c():
    if False:
        return 10
    with client:
        response = client.get('/b/a/c')
    assert response.content == b'Hello B A C'
    assert response.headers['content-type'] == html_type

def test_router_b_a_c_override():
    if False:
        for i in range(10):
            print('nop')
    with client:
        response = client.get('/b/a/c/override')
    assert response.json() == {'msg': 'Hello B A C'}
    assert response.headers['content-type'] == override_type