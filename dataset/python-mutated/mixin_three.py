from typing import Callable
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import deferred
from sqlalchemy.orm import Mapped
from sqlalchemy.orm.decl_api import declarative_mixin
from sqlalchemy.orm.decl_api import declared_attr
from sqlalchemy.orm.interfaces import MapperProperty

def some_other_decorator(fn: Callable[..., None]) -> Callable[..., None]:
    if False:
        i = 10
        return i + 15
    return fn

@declarative_mixin
class HasAMixin:
    x: Mapped[int] = Column(Integer)
    y = Column(String)

    @declared_attr
    def data(cls) -> Column[String]:
        if False:
            while True:
                i = 10
        return Column(String)

    @declared_attr
    def data2(cls) -> MapperProperty[str]:
        if False:
            print('Hello World!')
        return deferred(Column(String))

    @some_other_decorator
    def q(cls) -> None:
        if False:
            return 10
        return None