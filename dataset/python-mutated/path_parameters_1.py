from pydantic import BaseModel
from litestar import Litestar, get
USER_DB = {1: {'id': 1, 'name': 'John Doe'}}

class User(BaseModel):
    id: int
    name: str

@get('/user/{user_id:int}', sync_to_thread=False)
def get_user(user_id: int) -> User:
    if False:
        while True:
            i = 10
    return User.parse_obj(USER_DB[user_id])
app = Litestar(route_handlers=[get_user])