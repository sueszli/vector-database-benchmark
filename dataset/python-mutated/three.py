from pyramid.response import Response

def aview(request):
    if False:
        while True:
            i = 10
    return Response('three')

def configure(config):
    if False:
        print('Hello World!')
    config.add_view(aview, name='three')
    config.include('tests.pkgs.includeapp1.two.configure')
    config.add_view(aview)