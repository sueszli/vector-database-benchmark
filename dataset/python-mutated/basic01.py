from ninja.security import HttpBasicAuth

class BasicAuth(HttpBasicAuth):

    def authenticate(self, request, username, password):
        if False:
            i = 10
            return i + 15
        if username == 'admin' and password == 'secret':
            return username

@api.get('/basic', auth=BasicAuth())
def basic(request):
    if False:
        i = 10
        return i + 15
    return {'httpuser': request.auth}