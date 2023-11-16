from __future__ import annotations
from uuid import UUID
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import Annotated
from litestar import Litestar, put
from litestar.contrib.sqlalchemy.dto import SQLAlchemyDTO
from litestar.dto import DTOConfig
from .my_lib import Base

class A(Base):
    b_id: Mapped[UUID] = mapped_column(ForeignKey('b.id'))
    b: Mapped[B] = relationship(back_populates='a')

class B(Base):
    a: Mapped[A] = relationship(back_populates='b')
data_config = DTOConfig(max_nested_depth=0)
DataDTO = SQLAlchemyDTO[Annotated[A, data_config]]
ReturnDTO = SQLAlchemyDTO[A]

@put('/a', dto=DataDTO, return_dto=ReturnDTO, sync_to_thread=False)
def update_a(data: A) -> A:
    if False:
        i = 10
        return i + 15
    assert 'b' not in vars(data)
    data.b = B(id=data.b_id, a=data)
    return data
app = Litestar(route_handlers=[update_a])