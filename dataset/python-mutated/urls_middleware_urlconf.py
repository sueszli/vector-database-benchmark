from django.http import HttpResponse
from django.urls import path

def empty_response(request):
    if False:
        while True:
            i = 10
    return HttpResponse()
urlpatterns = [path('middleware_urlconf_view/', empty_response, name='middleware_urlconf_view')]