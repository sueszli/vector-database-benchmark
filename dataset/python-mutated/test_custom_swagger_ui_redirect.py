from fastapi import FastAPI
from fastapi.testclient import TestClient
swagger_ui_oauth2_redirect_url = '/docs/redirect'
app = FastAPI(swagger_ui_oauth2_redirect_url=swagger_ui_oauth2_redirect_url)

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
    assert f"oauth2RedirectUrl: window.location.origin + '{swagger_ui_oauth2_redirect_url}'" in response.text

def test_swagger_ui_oauth2_redirect():
    if False:
        i = 10
        return i + 15
    response = client.get(swagger_ui_oauth2_redirect_url)
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'text/html; charset=utf-8'
    assert 'window.opener.swaggerUIRedirectOauth2' in response.text

def test_response():
    if False:
        print('Hello World!')
    response = client.get('/items/')
    assert response.json() == {'id': 'foo'}