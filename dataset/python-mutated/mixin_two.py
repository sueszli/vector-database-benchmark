from typing import Callable
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import deferred
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.orm.decl_api import declared_attr
from sqlalchemy.orm.interfaces import MapperProperty
from sqlalchemy.sql.schema import ForeignKey
reg: registry = registry()

@reg.mapped
class C:
    __tablename__ = 'c'
    id = Column(Integer, primary_key=True)

def some_other_decorator(fn: Callable[..., None]) -> Callable[..., None]:
    if False:
        print('Hello World!')
    return fn

class HasAMixin:

    @declared_attr
    def a(cls) -> Mapped['A']:
        if False:
            while True:
                i = 10
        return relationship('A', back_populates='bs')

    @declared_attr
    def a2(cls):
        if False:
            return 10
        return relationship('A', back_populates='bs')

    @declared_attr
    def a3(cls) -> RelationshipProperty['A']:
        if False:
            for i in range(10):
                print('nop')
        return relationship('A', back_populates='bs')

    @declared_attr
    def c1(cls) -> RelationshipProperty[C]:
        if False:
            i = 10
            return i + 15
        return relationship(C, back_populates='bs')

    @declared_attr
    def c2(cls) -> Mapped[C]:
        if False:
            print('Hello World!')
        return relationship(C, back_populates='bs')

    @declared_attr
    def data(cls) -> Column[String]:
        if False:
            while True:
                i = 10
        return Column(String)

    @declared_attr
    def data2(cls) -> MapperProperty[str]:
        if False:
            return 10
        return deferred(Column(String))

    @some_other_decorator
    def q(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        return None

@reg.mapped
class B(HasAMixin):
    __tablename__ = 'b'
    id = Column(Integer, primary_key=True)
    a_id: int = Column(ForeignKey('a.id'))
    c_id: int = Column(ForeignKey('c.id'))

@reg.mapped
class A:
    __tablename__ = 'a'
    id = Column(Integer, primary_key=True)

    @declared_attr
    def data(cls) -> Column[String]:
        if False:
            for i in range(10):
                print('nop')
        return Column(String)

    @declared_attr
    def data2(cls):
        if False:
            return 10
        return Column(String)
    bs = relationship(B, uselist=True, back_populates='a')
a1 = A(id=1, data='d1', data2='d2')
b1 = B(a=A(), a2=A(), c1=C(), c2=C(), data='d1', data2='d2')
B.a.any()
B.a2.any()
B.c1.any()
B.c2.any()
B.q.any()
B.data.in_(['a', 'b'])
B.data2.in_(['a', 'b'])