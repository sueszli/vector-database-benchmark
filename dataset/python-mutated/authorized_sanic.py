from functools import wraps
from sanic import Sanic
from sanic.response import json
app = Sanic('Example')

def check_request_for_authorization_status(request):
    if False:
        return 10
    flag = True
    return flag

def authorized(f):
    if False:
        return 10

    @wraps(f)
    async def decorated_function(request, *args, **kwargs):
        is_authorized = check_request_for_authorization_status(request)
        if is_authorized:
            response = await f(request, *args, **kwargs)
            return response
        else:
            return json({'status': 'not_authorized'}, 403)
    return decorated_function

@app.route('/')
@authorized
async def test(request):
    return json({'status': 'authorized'})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)