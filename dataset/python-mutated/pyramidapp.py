from pyramid.config import Configurator
from pyramid.response import Response

def hello_world(request):
    if False:
        return 10
    return Response('Hello world!')

def goodbye_world(request):
    if False:
        while True:
            i = 10
    return Response('Goodbye world!')
config = Configurator()
config.add_view(hello_world)
config.add_view(goodbye_world, name='goodbye')
app = config.make_wsgi_app()