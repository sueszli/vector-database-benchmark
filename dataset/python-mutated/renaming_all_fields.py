from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import Annotated
from litestar import Litestar, post
from litestar.contrib.sqlalchemy.dto import SQLAlchemyDTO
from litestar.dto import DTOConfig, dto_field
from .my_lib import Base

class User(Base):
    first_name: Mapped[str]
    password: Mapped[str] = mapped_column(info=dto_field('private'))
    created_at: Mapped[datetime] = mapped_column(info=dto_field('read-only'))
config = DTOConfig(rename_strategy='camel')
UserDTO = SQLAlchemyDTO[Annotated[User, config]]

@post('/users', dto=UserDTO, sync_to_thread=False)
def create_user(data: User) -> User:
    if False:
        i = 10
        return i + 15
    assert data.first_name == 'Litestar User'
    data.created_at = datetime.min
    return data
app = Litestar(route_handlers=[create_user])