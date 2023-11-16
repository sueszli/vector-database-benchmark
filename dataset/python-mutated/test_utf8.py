from json import dumps as json_dumps
from sanic.response import text

def test_utf8_query_string(app):
    if False:
        return 10

    @app.route('/')
    async def handler(request):
        return text('OK')
    (request, response) = app.test_client.get('/', params=[('utf8', '✓')])
    assert request.args.get('utf8') == '✓'

def test_utf8_response(app):
    if False:
        print('Hello World!')

    @app.route('/')
    async def handler(request):
        return text('✓')
    (request, response) = app.test_client.get('/')
    assert response.text == '✓'

def skip_test_utf8_route(app):
    if False:
        i = 10
        return i + 15

    @app.route('/')
    async def handler(request):
        return text('OK')
    (request, response) = app.test_client.get('/✓')
    assert response.text == 'OK'

def test_utf8_post_json(app):
    if False:
        while True:
            i = 10

    @app.post('/')
    async def handler(request):
        return text('OK')
    payload = {'test': '✓'}
    headers = {'content-type': 'application/json'}
    (request, response) = app.test_client.post('/', data=json_dumps(payload), headers=headers)
    assert request.json.get('test') == '✓'
    assert response.text == 'OK'