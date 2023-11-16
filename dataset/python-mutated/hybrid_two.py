from __future__ import annotations
import typing
from sqlalchemy import Float
from sqlalchemy import func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.sql.expression import ColumnElement

class Base(DeclarativeBase):
    pass

class Interval(Base):
    __tablename__ = 'interval'
    id: Mapped[int] = mapped_column(primary_key=True)
    start: Mapped[int]
    end: Mapped[int]

    def __init__(self, start: int, end: int):
        if False:
            print('Hello World!')
        self.start = start
        self.end = end

    @hybrid_property
    def length(self) -> int:
        if False:
            while True:
                i = 10
        return self.end - self.start

    @hybrid_property
    def _inst_radius(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return abs(self.length) / 2

    @_inst_radius.expression
    def old_radius(cls) -> ColumnElement[float]:
        if False:
            for i in range(10):
                print('nop')
        f1 = func.abs(cls.length, type_=Float())
        expr = f1 / 2
        if typing.TYPE_CHECKING:
            reveal_type(f1)
            reveal_type(expr)
        return expr

    @hybrid_property
    def new_radius(self) -> float:
        if False:
            while True:
                i = 10
        return abs(self.length) / 2

    @new_radius.inplace.expression
    @classmethod
    def _new_radius_expr(cls) -> ColumnElement[float]:
        if False:
            while True:
                i = 10
        f1 = func.abs(cls.length, type_=Float())
        expr = f1 / 2
        if typing.TYPE_CHECKING:
            reveal_type(f1)
            reveal_type(expr)
        return expr
i1 = Interval(5, 10)
i2 = Interval(7, 12)
l1: int = i1.length
rdo: float = i2.old_radius
rdn: float = i2.new_radius
expr1 = Interval.length.in_([5, 10])
expr2o = Interval.old_radius
expr2n = Interval.new_radius
expr3o = Interval.old_radius.in_([0.5, 5.2])
expr3n = Interval.new_radius.in_([0.5, 5.2])
if typing.TYPE_CHECKING:
    reveal_type(i1.length)
    reveal_type(i2.old_radius)
    reveal_type(i2.new_radius)
    reveal_type(Interval.length)
    reveal_type(Interval.old_radius)
    reveal_type(Interval.new_radius)
    reveal_type(expr1)
    reveal_type(expr2o)
    reveal_type(expr2n)
    reveal_type(expr3o)
    reveal_type(expr3n)

class Foo(Base):
    val: bool

    def needs_update_getter(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.val
        ...

    def needs_update_setter(self, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.val = value
    needs_update: hybrid_property[bool] = hybrid_property(needs_update_getter, needs_update_setter)