from webob import Response

def bview(request):
    if False:
        i = 10
        return i + 15
    return Response('b view')

def includeme(config):
    if False:
        print('Hello World!')
    config.add_view(bview)