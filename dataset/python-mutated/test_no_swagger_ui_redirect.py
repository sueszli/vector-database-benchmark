from fastapi import FastAPI
from fastapi.testclient import TestClient
app = FastAPI(swagger_ui_oauth2_redirect_url=None)

@app.get('/items/')
async def read_items():
    return {'id': 'foo'}
client = TestClient(app)

def test_swagger_ui():
    if False:
        return 10
    response = client.get('/docs')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'text/html; charset=utf-8'
    assert 'swagger-ui-dist' in response.text
    print(client.base_url)
    assert 'oauth2RedirectUrl' not in response.text

def test_swagger_ui_no_oauth2_redirect():
    if False:
        return 10
    response = client.get('/docs/oauth2-redirect')
    assert response.status_code == 404, response.text

def test_response():
    if False:
        print('Hello World!')
    response = client.get('/items/')
    assert response.json() == {'id': 'foo'}