from os import environ
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr
from litestar import Litestar, Request, Response, get, post
from litestar.connection import ASGIConnection
from litestar.contrib.jwt import JWTCookieAuth, Token
from litestar.openapi.config import OpenAPIConfig

class User(BaseModel):
    id: UUID
    name: str
    email: EmailStr
MOCK_DB: Dict[str, User] = {}

async def retrieve_user_handler(token: 'Token', connection: 'ASGIConnection[Any, Any, Any, Any]') -> Optional[User]:
    return MOCK_DB.get(token.sub)
jwt_cookie_auth = JWTCookieAuth[User](retrieve_user_handler=retrieve_user_handler, token_secret=environ.get('JWT_SECRET', 'abcd123'), exclude=['/login', '/schema'])

@post('/login')
async def login_handler(data: 'User') -> 'Response[User]':
    MOCK_DB[str(data.id)] = data
    return jwt_cookie_auth.login(identifier=str(data.id), response_body=data)

@get('/some-path', sync_to_thread=False)
def some_route_handler(request: 'Request[User, Token, Any]') -> Any:
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(request.user, User)
    assert isinstance(request.auth, Token)
openapi_config = OpenAPIConfig(title='My API', version='1.0.0')
app = Litestar(route_handlers=[login_handler, some_route_handler], on_app_init=[jwt_cookie_auth.on_app_init], openapi_config=openapi_config)