from sanic_testing.testing import PORT, SanicTestClient
from sanic.response import json, text

def test_test_client_port_none(app):
    if False:
        for i in range(10):
            print('nop')

    @app.get('/get')
    def handler(request):
        if False:
            return 10
        return text('OK')
    test_client = SanicTestClient(app, port=None)
    (request, response) = test_client.get('/get')
    assert response.text == 'OK'
    (request, response) = test_client.post('/get')
    assert response.status == 405

def test_test_client_port_default(app):
    if False:
        for i in range(10):
            print('nop')

    @app.get('/get')
    def handler(request):
        if False:
            print('Hello World!')
        return json(request.transport.get_extra_info('sockname')[1])
    test_client = SanicTestClient(app)
    assert test_client.port == PORT
    (request, response) = test_client.get('/get')
    assert test_client.port > 0
    assert response.json == test_client.port