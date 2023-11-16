from __future__ import annotations
import typing
from sqlalchemy import select
from sqlalchemy.ext.hybrid import hybrid_method
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

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
            print('Hello World!')
        return self.end - self.start

    @hybrid_method
    def contains(self, point: int) -> int:
        if False:
            i = 10
            return i + 15
        return (self.start <= point) & (point <= self.end)

    @hybrid_method
    def intersects(self, other: Interval) -> int:
        if False:
            print('Hello World!')
        return self.contains(other.start) | self.contains(other.end)

    @hybrid_method
    def fancy_thing(self, point: int, x: int, y: int) -> bool:
        if False:
            return 10
        return (self.start <= point) & (point <= self.end)
i1 = Interval(5, 10)
i2 = Interval(7, 12)
expr1 = Interval.length.in_([5, 10])
expr2 = Interval.contains(7)
expr3 = Interval.intersects(i2)
expr4 = Interval.fancy_thing(10, 12, 15)
Interval.fancy_thing(1, 2)
Interval.fancy_thing(1, 'foo', 3)
stmt1 = select(Interval).where(expr1).where(expr4)
stmt2 = select(expr4)
if typing.TYPE_CHECKING:
    reveal_type(i1.length)
    reveal_type(Interval.length)
    reveal_type(expr1)
    reveal_type(expr2)
    reveal_type(expr3)
    reveal_type(i1.fancy_thing(1, 2, 3))
    reveal_type(expr4)
    reveal_type(stmt2)