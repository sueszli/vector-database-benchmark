from typing import Dict, Optional
from django.http import HttpRequest, HttpResponse

def test_index(request: HttpRequest):
    if False:
        print('Hello World!')
    eval(request.GET['bad'])

def test_get(request: HttpRequest):
    if False:
        return 10
    eval(request.GET.get('bad'))

def test_getlist(request: HttpRequest):
    if False:
        i = 10
        return i + 15
    eval(request.GET.getlist('bad'))

def test_optional(request: Optional[HttpRequest]):
    if False:
        print('Hello World!')
    eval(request.GET['bad'])

def test_assigning_to_request_fields(request: HttpRequest):
    if False:
        for i in range(10):
            print('nop')
    request.GET = request.GET.copy()
    eval(request.GET['bad'])
    request.POST = request.POST.copy()
    eval(request.POST['bad'])