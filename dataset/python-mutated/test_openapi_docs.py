from django.conf import settings
from django.test import override_settings
from ninja import NinjaAPI, Redoc, Swagger
from ninja.testing import TestClient
NO_NINJA_INSTALLED_APPS = [i for i in settings.INSTALLED_APPS if i != 'ninja']

def test_swagger():
    if False:
        for i in range(10):
            print('nop')
    'Default engine is swagger'
    api = NinjaAPI()
    assert isinstance(api.docs, Swagger)
    client = TestClient(api)
    response = client.get('/docs')
    assert response.status_code == 200
    assert b'swagger-ui-init.js' in response.content

    @override_settings(INSTALLED_APPS=NO_NINJA_INSTALLED_APPS)
    def call_docs():
        if False:
            i = 10
            return i + 15
        response = client.get('/docs')
        assert response.status_code == 200
        assert b'https://cdn.jsdelivr.net/npm/swagger-ui-dist' in response.content
    call_docs()

def test_swagger_settings():
    if False:
        while True:
            i = 10
    api = NinjaAPI(docs=Swagger(settings={'persistAuthorization': True}))
    client = TestClient(api)
    response = client.get('/docs')
    assert response.status_code == 200
    assert b'"persistAuthorization": true' in response.content

def test_redoc():
    if False:
        return 10
    api = NinjaAPI(docs=Redoc())
    client = TestClient(api)
    response = client.get('/docs')
    assert response.status_code == 200
    assert b'redoc.standalone.js' in response.content

    @override_settings(INSTALLED_APPS=NO_NINJA_INSTALLED_APPS)
    def call_docs():
        if False:
            while True:
                i = 10
        response = client.get('/docs')
        assert response.status_code == 200
        assert b'https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js' in response.content
    call_docs()

def test_redoc_settings():
    if False:
        print('Hello World!')
    api = NinjaAPI(docs=Redoc(settings={'disableSearch': True}))
    client = TestClient(api)
    response = client.get('/docs')
    assert response.status_code == 200
    assert b'"disableSearch": true' in response.content