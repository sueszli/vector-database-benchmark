from fastapi import APIRouter
ping_router = APIRouter(prefix='/ping', tags=['ping'])

@ping_router.get('/')
def pong():
    if False:
        while True:
            i = 10
    return {'message': 'pong!'}