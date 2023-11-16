from ninja.security import APIKeyQuery, APIKeyHeader

class AuthCheck:

    def authenticate(self, request, key):
        if False:
            print('Hello World!')
        if key == 'supersecret':
            return key

class QueryKey(AuthCheck, APIKeyQuery):
    pass

class HeaderKey(AuthCheck, APIKeyHeader):
    pass

@api.get('/multiple', auth=[QueryKey(), HeaderKey()])
def multiple(request):
    if False:
        for i in range(10):
            print('nop')
    return f'Token = {request.auth}'