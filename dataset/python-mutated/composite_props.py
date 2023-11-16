from typing import Any
from typing import Tuple
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import composite
Base = declarative_base()

class Point:

    def __init__(self, x: int, y: int):
        if False:
            print('Hello World!')
        self.x = x
        self.y = y

    def __composite_values__(self) -> Tuple[int, int]:
        if False:
            print('Hello World!')
        return (self.x, self.y)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Point(x=%r, y=%r)' % (self.x, self.y)

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, Point) and other.x == self.x and (other.y == self.y)

    def __ne__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

class Vertex(Base):
    __tablename__ = 'vertices'
    id = Column(Integer, primary_key=True)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    start = composite(Point, x1, y1)
    end: Point = composite(Point, x2, y2)
v1 = Vertex(start=Point(3, 4), end=Point(5, 6))
stmt = select(Vertex).where(Vertex.start.in_([Point(3, 4)]))
p1: Point = v1.start
p2: Point = v1.end
y3: int = v1.end.y