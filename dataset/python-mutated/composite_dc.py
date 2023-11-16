import dataclasses
from sqlalchemy import select
from sqlalchemy.orm import composite
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass

@dataclasses.dataclass
class Point:

    def __init__(self, x: int, y: int):
        if False:
            return 10
        self.x = x
        self.y = y

class Vertex(Base):
    __tablename__ = 'vertices'
    id: Mapped[int] = mapped_column(primary_key=True)
    x1: Mapped[int]
    y1: Mapped[int]
    x2: Mapped[int]
    y2: Mapped[int]
    start = composite(Point, 'x1', 'y1')
    end: Mapped[Point] = composite(Point, 'x2', 'y2')
v1 = Vertex(start=Point(3, 4), end=Point(5, 6))
stmt = select(Vertex).where(Vertex.start.in_([Point(3, 4)]))
reveal_type(stmt)
reveal_type(v1.start)
reveal_type(v1.end)
reveal_type(v1.end.y)