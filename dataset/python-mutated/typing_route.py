from __future__ import annotations
import typing as t
from http import HTTPStatus
from flask import Flask
from flask import jsonify
from flask import stream_template
from flask.templating import render_template
from flask.views import View
from flask.wrappers import Response
app = Flask(__name__)

@app.route('/str')
def hello_str() -> str:
    if False:
        for i in range(10):
            print('nop')
    return '<p>Hello, World!</p>'

@app.route('/bytes')
def hello_bytes() -> bytes:
    if False:
        return 10
    return b'<p>Hello, World!</p>'

@app.route('/json')
def hello_json() -> Response:
    if False:
        print('Hello World!')
    return jsonify('Hello, World!')

@app.route('/json/dict')
def hello_json_dict() -> dict[str, t.Any]:
    if False:
        while True:
            i = 10
    return {'response': 'Hello, World!'}

@app.route('/json/dict')
def hello_json_list() -> list[t.Any]:
    if False:
        i = 10
        return i + 15
    return [{'message': 'Hello'}, {'message': 'World'}]

class StatusJSON(t.TypedDict):
    status: str

@app.route('/typed-dict')
def typed_dict() -> StatusJSON:
    if False:
        return 10
    return {'status': 'ok'}

@app.route('/generator')
def hello_generator() -> t.Generator[str, None, None]:
    if False:
        return 10

    def show() -> t.Generator[str, None, None]:
        if False:
            while True:
                i = 10
        for x in range(100):
            yield f'data:{x}\n\n'
    return show()

@app.route('/generator-expression')
def hello_generator_expression() -> t.Iterator[bytes]:
    if False:
        print('Hello World!')
    return (f'data:{x}\n\n'.encode() for x in range(100))

@app.route('/iterator')
def hello_iterator() -> t.Iterator[str]:
    if False:
        return 10
    return iter([f'data:{x}\n\n' for x in range(100)])

@app.route('/status')
@app.route('/status/<int:code>')
def tuple_status(code: int=200) -> tuple[str, int]:
    if False:
        for i in range(10):
            print('nop')
    return ('hello', code)

@app.route('/status-enum')
def tuple_status_enum() -> tuple[str, int]:
    if False:
        while True:
            i = 10
    return ('hello', HTTPStatus.OK)

@app.route('/headers')
def tuple_headers() -> tuple[str, dict[str, str]]:
    if False:
        print('Hello World!')
    return ('Hello, World!', {'Content-Type': 'text/plain'})

@app.route('/template')
@app.route('/template/<name>')
def return_template(name: str | None=None) -> str:
    if False:
        return 10
    return render_template('index.html', name=name)

@app.route('/template')
def return_template_stream() -> t.Iterator[str]:
    if False:
        i = 10
        return i + 15
    return stream_template('index.html', name='Hello')

@app.route('/async')
async def async_route() -> str:
    return 'Hello'

class RenderTemplateView(View):

    def __init__(self: RenderTemplateView, template_name: str) -> None:
        if False:
            print('Hello World!')
        self.template_name = template_name

    def dispatch_request(self: RenderTemplateView) -> str:
        if False:
            print('Hello World!')
        return render_template(self.template_name)
app.add_url_rule('/about', view_func=RenderTemplateView.as_view('about_page', template_name='about.html'))