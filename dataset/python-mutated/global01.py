from ninja import NinjaAPI, Form
from ninja.security import HttpBearer

class GlobalAuth(HttpBearer):

    def authenticate(self, request, token):
        if False:
            while True:
                i = 10
        if token == 'supersecret':
            return token
api = NinjaAPI(auth=GlobalAuth())

@api.post('/token', auth=None)
def get_token(request, username: str=Form(...), password: str=Form(...)):
    if False:
        print('Hello World!')
    if username == 'admin' and password == 'giraffethinnknslong':
        return {'token': 'supersecret'}