import inspect
import os
import sys
from sanic import Sanic
from sanic.exceptions import ServerError
from sanic.response import json, text
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + '/../../../')
app = Sanic('test')

@app.route('/')
async def test(request):
    return json({'test': True})

@app.route('/sync', methods=['GET', 'POST'])
def test(request):
    if False:
        for i in range(10):
            print('nop')
    return json({'test': True})

@app.route('/text/<name>/<butt:int>')
def rtext(request, name, butt):
    if False:
        print('Hello World!')
    return text('yeehaww {} {}'.format(name, butt))

@app.route('/exception')
def exception(request):
    if False:
        for i in range(10):
            print('nop')
    raise ServerError('yep')

@app.route('/exception/async')
async def test(request):
    raise ServerError('asunk')

@app.route('/post_json')
def post_json(request):
    if False:
        for i in range(10):
            print('nop')
    return json({'received': True, 'message': request.json})

@app.route('/query_string')
def query_string(request):
    if False:
        return 10
    return json({'parsed': True, 'args': request.args, 'url': request.url, 'query_string': request.query_string})
app.run(host='0.0.0.0', port=sys.argv[1])