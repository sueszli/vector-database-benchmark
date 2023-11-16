"""Tests Mapped covariance."""
from datetime import datetime
from typing import Protocol
from typing import Union
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Nullable
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.sql.elements import SQLCoreOperations

class ParentProtocol(Protocol):
    name: Mapped[str]

class ChildProtocol(Protocol):

    @property
    def parent(self) -> Mapped[ParentProtocol]:
        if False:
            for i in range(10):
                print('nop')
        ...

def get_parent_name(child: ChildProtocol) -> str:
    if False:
        print('Hello World!')
    return child.parent.name

class Base(DeclarativeBase):
    pass

class Parent(Base):
    __tablename__ = 'parent'
    name: Mapped[str] = mapped_column(primary_key=True)

class Child(Base):
    __tablename__ = 'child'
    name: Mapped[str] = mapped_column(primary_key=True)
    parent_name: Mapped[str] = mapped_column(ForeignKey(Parent.name))
    parent: Mapped[Parent] = relationship()
assert get_parent_name(Child(parent=Parent(name='foo'))) == 'foo'

class NullableModel(DeclarativeBase):
    not_null: Mapped[datetime]
    nullable: Mapped[Union[datetime, None]]
test = NullableModel()
test.not_null = func.now()
test.nullable = func.now()
nullable_now: SQLCoreOperations[Union[datetime, None]] = Nullable(func.now())
test.nullable = Nullable(func.now())