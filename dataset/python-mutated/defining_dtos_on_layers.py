from typing import List
from uuid import UUID, uuid4
from litestar import Controller, delete, get, post, put
from .models import User, UserDTO, UserReturnDTO

class UserController(Controller):
    dto = UserDTO
    return_dto = UserReturnDTO

    @post()
    def create_user(self, data: User) -> User:
        if False:
            print('Hello World!')
        return data

    @get()
    def get_users(self) -> List[User]:
        if False:
            print('Hello World!')
        return [User(id=uuid4(), name='Litestar User')]

    @get('/{user_id:uuid}')
    def get_user(self, user_id: UUID) -> User:
        if False:
            for i in range(10):
                print('nop')
        return User(id=user_id, name='Litestar User')

    @put('/{user_id:uuid}')
    def update_user(self, data: User) -> User:
        if False:
            print('Hello World!')
        return data

    @delete('/{user_id:uuid}', return_dto=None)
    def delete_user(self, user_id: UUID) -> None:
        if False:
            for i in range(10):
                print('nop')
        return None