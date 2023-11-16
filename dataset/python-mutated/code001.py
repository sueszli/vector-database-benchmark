from ninja import NinjaAPI
from ninja.security import django_auth
api = NinjaAPI(csrf=True)

@api.get('/pets', auth=django_auth)
def pets(request):
    if False:
        i = 10
        return i + 15
    return f'Authenticated user {request.auth}'