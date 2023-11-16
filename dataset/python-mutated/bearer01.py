from ninja.security import HttpBearer

class AuthBearer(HttpBearer):

    def authenticate(self, request, token):
        if False:
            print('Hello World!')
        if token == 'supersecret':
            return token

@api.get('/bearer', auth=AuthBearer())
def bearer(request):
    if False:
        print('Hello World!')
    return {'token': request.auth}