from django.http import HttpResponse
from ninja import NinjaAPI
from ninja.testing import TestClient
api = NinjaAPI()

@api.get('/test-no-cookies')
def op_no_cookies(request):
    if False:
        while True:
            i = 10
    return {}

@api.get('/test-set-cookie')
def op_set_cookie(request):
    if False:
        return 10
    response = HttpResponse()
    response.set_cookie(key='sessionid', value='sessionvalue')
    return response
client = TestClient(api)

def test_cookies():
    if False:
        print('Hello World!')
    assert bool(client.get('/test-no-cookies').cookies) is False
    assert 'sessionid' in client.get('/test-set-cookie').cookies