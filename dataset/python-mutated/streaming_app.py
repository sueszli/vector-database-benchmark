from werkzeug.wrappers import Request
from werkzeug.wrappers import Response

@Request.application
def app(request):
    if False:
        return 10

    def gen():
        if False:
            print('Hello World!')
        for x in range(5):
            yield f'{x}\n'
        if request.path == '/crash':
            raise Exception('crash requested')
    return Response(gen())