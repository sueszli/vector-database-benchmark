from random import randint
from sanic import Sanic
from sanic.response import text
app = Sanic('Example')

@app.middleware('request')
def append_request(request):
    if False:
        while True:
            i = 10
    request.ctx.num = randint(0, 100)

@app.get('/pop')
def pop_handler(request):
    if False:
        return 10
    return text(request.ctx.num)

@app.get('/key_exist')
def key_exist_handler(request):
    if False:
        while True:
            i = 10
    if hasattr(request.ctx, 'num'):
        return text('num exist in request')
    return text('num does not exist in request')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)