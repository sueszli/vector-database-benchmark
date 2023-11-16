import pytest
from sanic_routing.exceptions import RouteExists
from sanic import Sanic
from sanic.response import text

def test_vhosts():
    if False:
        while True:
            i = 10
    app = Sanic('app')

    @app.route('/', host='example.com')
    async def handler1(request):
        return text("You're at example.com!")

    @app.route('/', host='subdomain.example.com')
    async def handler2(request):
        return text("You're at subdomain.example.com!")
    headers = {'Host': 'example.com'}
    (request, response) = app.test_client.get('/', headers=headers)
    assert response.text == "You're at example.com!"
    headers = {'Host': 'subdomain.example.com'}
    (request, response) = app.test_client.get('/', headers=headers)
    assert response.text == "You're at subdomain.example.com!"

def test_vhosts_with_list(app):
    if False:
        i = 10
        return i + 15

    @app.route('/', host=['hello.com', 'world.com'])
    async def handler(request):
        return text('Hello, world!')
    headers = {'Host': 'hello.com'}
    (request, response) = app.test_client.get('/', headers=headers)
    assert response.text == 'Hello, world!'
    headers = {'Host': 'world.com'}
    (request, response) = app.test_client.get('/', headers=headers)
    assert response.text == 'Hello, world!'

def test_vhosts_with_defaults(app):
    if False:
        print('Hello World!')

    @app.route('/', host='hello.com')
    async def handler1(request):
        return text('Hello, world!')
    with pytest.raises(RouteExists):

        @app.route('/')
        async def handler2(request):
            return text('default')
    headers = {'Host': 'hello.com'}
    (request, response) = app.test_client.get('/', headers=headers)
    assert response.text == 'Hello, world!'