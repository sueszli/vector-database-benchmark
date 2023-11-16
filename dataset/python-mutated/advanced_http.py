from nameko.web.handlers import http
from werkzeug.wrappers import Response

class Service:
    name = 'advanced_http_service'

    @http('GET', '/privileged')
    def forbidden(self, request):
        if False:
            for i in range(10):
                print('nop')
        return (403, 'Forbidden')

    @http('GET', '/headers')
    def redirect(self, request):
        if False:
            i = 10
            return i + 15
        return (201, {'Location': 'https://www.example.com/widget/1'}, '')

    @http('GET', '/custom')
    def custom(self, request):
        if False:
            for i in range(10):
                print('nop')
        return Response('payload')