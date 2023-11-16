from django.http import HttpResponse

def sendfile(request, filename, **kwargs):
    if False:
        print('Hello World!')
    '\n    Dummy sendfile backend implementation.\n    '
    return HttpResponse('Dummy backend response')