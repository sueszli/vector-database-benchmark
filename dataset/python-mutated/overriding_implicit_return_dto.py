from litestar import post
from .models import User, UserDTO

@post(dto=UserDTO, return_dto=None)
def create_user(data: User) -> bytes:
    if False:
        print('Hello World!')
    return data.name.encode(encoding='utf-8')