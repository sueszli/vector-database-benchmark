from sanic import Blueprint, Sanic
from sanic.response import text
'\nDemonstrates that blueprint request middleware are executed in the order they\nare added. And blueprint response middleware are executed in _reverse_ order.\nOn a valid request, it should print "1 2 3 6 5 4" to terminal\n'
app = Sanic('Example')
bp = Blueprint('bp_example')

@bp.on_request
def request_middleware_1(request):
    if False:
        while True:
            i = 10
    print('1')

@bp.on_request
def request_middleware_2(request):
    if False:
        i = 10
        return i + 15
    print('2')

@bp.on_request
def request_middleware_3(request):
    if False:
        return 10
    print('3')

@bp.on_response
def resp_middleware_4(request, response):
    if False:
        return 10
    print('4')

@bp.on_response
def resp_middleware_5(request, response):
    if False:
        i = 10
        return i + 15
    print('5')

@bp.on_response
def resp_middleware_6(request, response):
    if False:
        return 10
    print('6')

@bp.route('/')
def pop_handler(request):
    if False:
        for i in range(10):
            print('nop')
    return text('hello world')
app.blueprint(bp, url_prefix='/bp')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, auto_reload=False)