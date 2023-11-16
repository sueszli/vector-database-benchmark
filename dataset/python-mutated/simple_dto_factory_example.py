from datetime import datetime
from sqlalchemy.orm import Mapped
from litestar import Litestar, post
from litestar.contrib.sqlalchemy.dto import SQLAlchemyDTO
from .my_lib import Base

class User(Base):
    name: Mapped[str]
    password: Mapped[str]
    created_at: Mapped[datetime]
UserDTO = SQLAlchemyDTO[User]

@post('/users', dto=UserDTO, sync_to_thread=False)
def create_user(data: User) -> User:
    if False:
        i = 10
        return i + 15
    return data
app = Litestar(route_handlers=[create_user])