from typing import Any, Dict, Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, EmailStr, SecretStr
from litestar import Litestar, Request, get, post
from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.middleware.session.server_side import ServerSideSessionBackend, ServerSideSessionConfig
from litestar.openapi.config import OpenAPIConfig
from litestar.security.session_auth import SessionAuth
from litestar.stores.memory import MemoryStore

class User(BaseModel):
    id: UUID
    name: str
    email: EmailStr

class UserCreatePayload(BaseModel):
    name: str
    email: EmailStr
    password: SecretStr

class UserLoginPayload(BaseModel):
    email: EmailStr
    password: SecretStr
MOCK_DB: Dict[str, User] = {}
memory_store = MemoryStore()

async def retrieve_user_handler(session: Dict[str, Any], connection: 'ASGIConnection[Any, Any, Any, Any]') -> Optional[User]:
    return MOCK_DB.get(user_id) if (user_id := session.get('user_id')) else None

@post('/login')
async def login(data: UserLoginPayload, request: 'Request[Any, Any, Any]') -> User:
    user_id = await memory_store.get(data.email)
    if not user_id:
        raise NotAuthorizedException
    user_id = user_id.decode('utf-8')
    request.set_session({'user_id': user_id})
    return MOCK_DB[user_id]

@post('/signup')
async def signup(data: UserCreatePayload, request: Request[Any, Any, Any]) -> User:
    user = User(name=data.name, email=data.email, id=uuid4())
    await memory_store.set(data.email, str(user.id))
    MOCK_DB[str(user.id)] = user
    request.set_session({'user_id': str(user.id)})
    return user

@get('/user', sync_to_thread=False)
def get_user(request: Request[User, Dict[Literal['user_id'], str], Any]) -> Any:
    if False:
        while True:
            i = 10
    return request.user
openapi_config = OpenAPIConfig(title='My API', version='1.0.0')
session_auth = SessionAuth[User, ServerSideSessionBackend](retrieve_user_handler=retrieve_user_handler, session_backend_config=ServerSideSessionConfig(), exclude=['/login', '/signup', '/schema'])
app = Litestar(route_handlers=[login, signup, get_user], on_app_init=[session_auth.on_app_init], openapi_config=openapi_config)