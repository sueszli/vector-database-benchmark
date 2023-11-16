from ninja import NinjaAPI, Router
from ninja.security import APIKeyQuery
from ninja.testing import TestClient

class KeyQuery1(APIKeyQuery):

    def authenticate(self, request, key):
        if False:
            for i in range(10):
                print('nop')
        if key == 'k1':
            return key

class KeyQuery2(APIKeyQuery):

    def authenticate(self, request, key):
        if False:
            while True:
                i = 10
        if key == 'k2':
            return key
api = NinjaAPI(auth=KeyQuery1())

@api.get('/default')
def default(request):
    if False:
        print('Hello World!')
    return {'auth': request.auth}

@api.api_operation(['POST', 'PATCH'], '/multi-method-no-auth')
def multi_no_auth(request):
    if False:
        for i in range(10):
            print('nop')
    return {'auth': request.auth}

@api.api_operation(['POST', 'PATCH'], '/multi-method-auth', auth=KeyQuery2())
def multi_auth(request):
    if False:
        print('Hello World!')
    return {'auth': request.auth}
router = Router()

@router.get('/router-operation')
def router_operation(request):
    if False:
        return 10
    return {'auth': str(request.auth)}

@router.get('/router-operation-auth', auth=KeyQuery2())
def router_operation_auth(request):
    if False:
        for i in range(10):
            print('nop')
    return {'auth': str(request.auth)}
api.add_router('', router)
client = TestClient(api)

def test_multi():
    if False:
        return 10
    assert client.get('/default').status_code == 401
    assert client.get('/default?key=k1').json() == {'auth': 'k1'}
    assert client.post('/multi-method-no-auth').status_code == 401
    assert client.post('/multi-method-no-auth?key=k1').json() == {'auth': 'k1'}
    assert client.patch('/multi-method-no-auth').status_code == 401
    assert client.patch('/multi-method-no-auth?key=k1').json() == {'auth': 'k1'}
    assert client.post('/multi-method-auth?key=k1').status_code == 401
    assert client.patch('/multi-method-auth?key=k1').status_code == 401
    assert client.post('/multi-method-auth?key=k2').json() == {'auth': 'k2'}
    assert client.patch('/multi-method-auth?key=k2').json() == {'auth': 'k2'}

def test_router_auth():
    if False:
        print('Hello World!')
    assert client.get('/router-operation').status_code == 401
    assert client.get('/router-operation?key=k1').json() == {'auth': 'k1'}
    assert client.get('/router-operation-auth?key=k1').status_code == 401
    assert client.get('/router-operation-auth?key=k2').json() == {'auth': 'k2'}