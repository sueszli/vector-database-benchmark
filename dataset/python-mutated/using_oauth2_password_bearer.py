from os import environ
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr
from litestar import Litestar, Request, Response, get, post
from litestar.connection import ASGIConnection
from litestar.contrib.jwt import OAuth2Login, OAuth2PasswordBearerAuth, Token
from litestar.openapi.config import OpenAPIConfig

class User(BaseModel):
    id: UUID
    name: str
    email: EmailStr
MOCK_DB: Dict[str, User] = {}

async def retrieve_user_handler(token: 'Token', connection: 'ASGIConnection[Any, Any, Any, Any]') -> Optional[User]:
    return MOCK_DB.get(token.sub)
oauth2_auth = OAuth2PasswordBearerAuth[User](retrieve_user_handler=retrieve_user_handler, token_secret=environ.get('JWT_SECRET', 'abcd123'), token_url='/login', exclude=['/login', '/schema'])

@post('/login')
async def login_handler(request: 'Request[Any, Any, Any]', data: 'User') -> 'Response[OAuth2Login]':
    MOCK_DB[str(data.id)] = data
    return oauth2_auth.login(identifier=str(data.id))

@post('/login_custom')
async def login_custom_response_handler(data: 'User') -> 'Response[User]':
    MOCK_DB[str(data.id)] = data
    return oauth2_auth.login(identifier=str(data.id), response_body=data)

@get('/some-path', sync_to_thread=False)
def some_route_handler(request: 'Request[User, Token, Any]') -> Any:
    if False:
        return 10
    assert isinstance(request.user, User)
    assert isinstance(request.auth, Token)
openapi_config = OpenAPIConfig(title='My API', version='1.0.0')
app = Litestar(route_handlers=[login_handler, some_route_handler], on_app_init=[oauth2_auth.on_app_init], openapi_config=openapi_config)