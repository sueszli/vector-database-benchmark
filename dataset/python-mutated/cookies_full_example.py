from os import urandom
from typing import Dict
from litestar import Litestar, Request, delete, get, post
from litestar.middleware.session.client_side import CookieBackendConfig
session_config = CookieBackendConfig(secret=urandom(16))

@get('/session', sync_to_thread=False)
def check_session_handler(request: Request) -> Dict[str, bool]:
    if False:
        print('Hello World!')
    'Handler function that accesses request.session.'
    return {'has_session': request.session != {}}

@post('/session', sync_to_thread=False)
def create_session_handler(request: Request) -> None:
    if False:
        return 10
    'Handler to set the session.'
    if not request.session:
        request.set_session({'username': 'moishezuchmir'})

@delete('/session', sync_to_thread=False)
def delete_session_handler(request: Request) -> None:
    if False:
        return 10
    'Handler to clear the session.'
    if request.session:
        request.clear_session()
app = Litestar(route_handlers=[check_session_handler, create_session_handler, delete_session_handler], middleware=[session_config.middleware])