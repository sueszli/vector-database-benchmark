from ninja import NinjaAPI
from ninja.security import HttpBearer
api = NinjaAPI()

class InvalidToken(Exception):
    pass

@api.exception_handler(InvalidToken)
def on_invalid_token(request, exc):
    if False:
        while True:
            i = 10
    return api.create_response(request, {'detail': 'Invalid token supplied'}, status=401)

class AuthBearer(HttpBearer):

    def authenticate(self, request, token):
        if False:
            while True:
                i = 10
        if token == 'supersecret':
            return token
        raise InvalidToken

@api.get('/bearer', auth=AuthBearer())
def bearer(request):
    if False:
        i = 10
        return i + 15
    return {'token': request.auth}