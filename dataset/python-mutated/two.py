from pyramid.response import Response

def aview(request):
    if False:
        print('Hello World!')
    return Response('two')

def configure(config):
    if False:
        while True:
            i = 10
    config.add_view(aview, name='two')
    config.include('tests.pkgs.includeapp1.three.configure')
    config.add_view(aview)