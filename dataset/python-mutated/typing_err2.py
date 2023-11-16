from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy.orm import RelationshipProperty
reg: registry = registry()

@reg.mapped
class Foo:
    id: int = Column(Integer())

    @declared_attr
    def name(cls) -> Column[str]:
        if False:
            i = 10
            return i + 15
        return Column(String)
    other_name: Column[String] = Column(String)

    @declared_attr
    def third_name(cls) -> Column:
        if False:
            while True:
                i = 10
        return Column(String)

    @declared_attr
    def some_relationship(cls) -> RelationshipProperty:
        if False:
            for i in range(10):
                print('nop')
        return relationship('Bar')
Foo(name='x')