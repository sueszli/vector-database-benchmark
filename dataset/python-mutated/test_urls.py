from django.http import HttpResponse, HttpResponseServerError
from django.urls import path
from django_oso import decorators
from django_oso.auth import authorize
from django_oso.decorators import authorize_request

def root(request):
    if False:
        while True:
            i = 10
    return HttpResponse('hello')

def auth(request):
    if False:
        while True:
            i = 10
    authorize(request, 'resource', action='read', actor='user')
    return HttpResponse('authorized')

@authorize_request(actor='user')
def auth_decorated_fail(request):
    if False:
        print('Hello World!')
    return HttpResponse('authorized')

@decorators.authorize(actor='user', action='read', resource='resource')
def auth_decorated(request):
    if False:
        for i in range(10):
            print('nop')
    return HttpResponse('authorized')

def a(request):
    if False:
        print('Hello World!')
    return HttpResponse('a')

def b(request):
    if False:
        return 10
    return HttpResponse('b')

def error(request):
    if False:
        i = 10
        return i + 15
    return HttpResponseServerError()
urlpatterns = [path('', root), path('auth/', auth), path('auth_decorated_fail/', auth_decorated_fail), path('auth_decorated/', auth_decorated), path('error/', error), path('a/', a), path('b/', b)]