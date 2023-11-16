from ninja.security import APIKeyHeader

class ApiKey(APIKeyHeader):
    param_name = 'X-API-Key'

    def authenticate(self, request, key):
        if False:
            while True:
                i = 10
        if key == 'supersecret':
            return key
header_key = ApiKey()

@api.get('/headerkey', auth=header_key)
def apikey(request):
    if False:
        print('Hello World!')
    return f'Token = {request.auth}'