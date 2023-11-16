from __future__ import annotations
from sqlalchemy import ForeignKey
from sqlalchemy import orm
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import selectinload

class Base(DeclarativeBase):
    pass

class A(Base):
    __tablename__ = 'a'
    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[str]
    bs: Mapped[list[B]] = relationship('B')

class B(Base):
    __tablename__ = 'b'
    id: Mapped[int] = mapped_column(primary_key=True)
    a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
    data: Mapped[str]
    a: Mapped[A] = relationship()

def test_9669_and() -> None:
    if False:
        while True:
            i = 10
    select(A).options(selectinload(A.bs.and_(B.data == 'some data')))

def test_9669_of_type() -> None:
    if False:
        print('Hello World!')
    ba = aliased(B)
    select(A).options(selectinload(A.bs.of_type(ba)))

def load_options_ok() -> None:
    if False:
        for i in range(10):
            print('nop')
    select(B).options(orm.contains_eager('*').contains_eager(A.bs), orm.load_only('*').load_only(A.bs), orm.joinedload('*').joinedload(A.bs), orm.subqueryload('*').subqueryload(A.bs), orm.selectinload('*').selectinload(A.bs), orm.lazyload('*').lazyload(A.bs), orm.immediateload('*').immediateload(A.bs), orm.noload('*').noload(A.bs), orm.raiseload('*').raiseload(A.bs), orm.defaultload('*').defaultload(A.bs), orm.defer('*').defer(A.bs), orm.undefer('*').undefer(A.bs))
    select(B).options(orm.contains_eager(B.a).contains_eager('*'), orm.load_only(B.a).load_only('*'), orm.joinedload(B.a).joinedload('*'), orm.subqueryload(B.a).subqueryload('*'), orm.selectinload(B.a).selectinload('*'), orm.lazyload(B.a).lazyload('*'), orm.immediateload(B.a).immediateload('*'), orm.noload(B.a).noload('*'), orm.raiseload(B.a).raiseload('*'), orm.defaultload(B.a).defaultload('*'), orm.defer(B.a).defer('*'), orm.undefer(B.a).undefer('*'))

def load_options_error() -> None:
    if False:
        return 10
    select(B).options(orm.contains_eager('foo'), orm.load_only('foo'), orm.joinedload('foo'), orm.subqueryload('foo'), orm.selectinload('foo'), orm.lazyload('foo'), orm.immediateload('foo'), orm.noload('foo'), orm.raiseload('foo'), orm.defaultload('foo'), orm.defer('foo'), orm.undefer('foo'))
    select(B).options(orm.contains_eager(B.a).contains_eager('bar'), orm.load_only(B.a).load_only('bar'), orm.joinedload(B.a).joinedload('bar'), orm.subqueryload(B.a).subqueryload('bar'), orm.selectinload(B.a).selectinload('bar'), orm.lazyload(B.a).lazyload('bar'), orm.immediateload(B.a).immediateload('bar'), orm.noload(B.a).noload('bar'), orm.raiseload(B.a).raiseload('bar'), orm.defaultload(B.a).defaultload('bar'), orm.defer(B.a).defer('bar'), orm.undefer(B.a).undefer('bar'))