from ninja.security import APIKeyCookie

class CookieKey(APIKeyCookie):

    def authenticate(self, request, key):
        if False:
            while True:
                i = 10
        if key == 'supersecret':
            return key
cookie_key = CookieKey()

@api.get('/cookiekey', auth=cookie_key)
def apikey(request):
    if False:
        print('Hello World!')
    return f'Token = {request.auth}'