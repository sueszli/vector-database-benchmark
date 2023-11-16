import os
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.testclient import TestClient
from apistar.client import Client, decoders
app = Starlette()

@app.route('/text-response/')
def text_response(request):
    if False:
        return 10
    return PlainTextResponse('hello, world')

@app.route('/file-response/')
def file_response(request):
    if False:
        i = 10
        return i + 15
    headers = {'Content-Type': 'image/png', 'Content-Disposition': 'attachment; filename="filename.png"'}
    return Response(b'<somedata>', headers=headers)

@app.route('/file-response-url-filename/name.png')
def file_response_url_filename(request):
    if False:
        i = 10
        return i + 15
    headers = {'Content-Type': 'image/png', 'Content-Disposition': 'attachment'}
    return Response(b'<somedata>', headers=headers)

@app.route('/file-response-no-extension/name')
def file_response_no_extension(request):
    if False:
        return 10
    headers = {'Content-Type': 'image/png', 'Content-Disposition': 'attachment'}
    return Response(b'<somedata>', headers=headers)

@app.route('/')
def file_response_no_name(request):
    if False:
        i = 10
        return i + 15
    headers = {'Content-Type': 'image/png', 'Content-Disposition': 'attachment'}
    return Response(b'<somedata>', headers=headers)
schema = {'openapi': '3.0.0', 'info': {'title': 'Test API', 'version': '1.0'}, 'servers': [{'url': 'http://testserver'}], 'paths': {'/text-response/': {'get': {'operationId': 'text-response'}}, '/file-response/': {'get': {'operationId': 'file-response'}}, '/file-response-url-filename/name.png': {'get': {'operationId': 'file-response-url-filename'}}, '/file-response-no-extension/name': {'get': {'operationId': 'file-response-no-extension'}}, '/': {'get': {'operationId': 'file-response-no-name'}}}}

def test_text_response():
    if False:
        while True:
            i = 10
    client = Client(schema, session=TestClient(app))
    data = client.request('text-response')
    assert data == 'hello, world'

def test_file_response():
    if False:
        for i in range(10):
            print('nop')
    client = Client(schema, session=TestClient(app))
    data = client.request('file-response')
    assert os.path.basename(data.name) == 'filename.png'
    assert data.read() == b'<somedata>'

def test_file_response_url_filename():
    if False:
        return 10
    client = Client(schema, session=TestClient(app))
    data = client.request('file-response-url-filename')
    assert os.path.basename(data.name) == 'name.png'
    assert data.read() == b'<somedata>'

def test_file_response_no_extension():
    if False:
        return 10
    client = Client(schema, session=TestClient(app))
    data = client.request('file-response-no-extension')
    assert os.path.basename(data.name) == 'name.png'
    assert data.read() == b'<somedata>'

def test_file_response_no_name():
    if False:
        print('Hello World!')
    client = Client(schema, session=TestClient(app))
    data = client.request('file-response-no-name')
    assert os.path.basename(data.name) == 'download.png'
    assert data.read() == b'<somedata>'

def test_unique_filename(tmpdir):
    if False:
        return 10
    client = Client(schema, session=TestClient(app), decoders=[decoders.DownloadDecoder(tmpdir)])
    data = client.request('file-response')
    assert os.path.basename(data.name) == 'filename.png'
    assert data.read() == b'<somedata>'
    data = client.request('file-response')
    assert os.path.basename(data.name) == 'filename (1).png'
    assert data.read() == b'<somedata>'