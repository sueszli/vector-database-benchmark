from litestar import post
from .models import User, UserDTO

@post(dto=UserDTO)
def create_user(data: User) -> User:
    if False:
        print('Hello World!')
    return data