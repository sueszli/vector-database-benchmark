import pytest
from sanic import Sanic, text

@pytest.fixture
def late_app(app: Sanic):
    if False:
        print('Hello World!')
    app.config.TOUCHUP = False
    app.get('/')(lambda _: text(''))
    return app

def test_late_route(late_app: Sanic):
    if False:
        return 10

    @late_app.before_server_start
    async def late(app: Sanic):

        @app.get('/late')
        def handler(_):
            if False:
                i = 10
                return i + 15
            return text('late')
    (_, response) = late_app.test_client.get('/late')
    assert response.status_code == 200
    assert response.text == 'late'

def test_late_middleware(late_app: Sanic):
    if False:
        while True:
            i = 10

    @late_app.get('/late')
    def handler(request):
        if False:
            for i in range(10):
                print('nop')
        return text(request.ctx.late)

    @late_app.before_server_start
    async def late(app: Sanic):

        @app.on_request
        def handler(request):
            if False:
                return 10
            request.ctx.late = 'late'
    (_, response) = late_app.test_client.get('/late')
    assert response.status_code == 200
    assert response.text == 'late'

def test_late_signal(late_app: Sanic):
    if False:
        i = 10
        return i + 15

    @late_app.get('/late')
    def handler(request):
        if False:
            i = 10
            return i + 15
        return text(request.ctx.late)

    @late_app.before_server_start
    async def late(app: Sanic):

        @app.signal('http.lifecycle.request')
        def handler(request):
            if False:
                return 10
            request.ctx.late = 'late'
    (_, response) = late_app.test_client.get('/late')
    assert response.status_code == 200
    assert response.text == 'late'