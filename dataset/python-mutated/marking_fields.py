from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from litestar import Litestar, post
from litestar.contrib.sqlalchemy.dto import SQLAlchemyDTO
from litestar.dto import dto_field
from .my_lib import Base

class User(Base):
    name: Mapped[str]
    password: Mapped[str] = mapped_column(info=dto_field('private'))
    created_at: Mapped[datetime] = mapped_column(info=dto_field('read-only'))
UserDTO = SQLAlchemyDTO[User]

@post('/users', dto=UserDTO, sync_to_thread=False)
def create_user(data: User) -> User:
    if False:
        print('Hello World!')
    assert 'password' not in vars(data)
    assert 'created_at' not in vars(data)
    data.created_at = datetime.min
    return data
app = Litestar(route_handlers=[create_user])