from typing import List
from typing import Optional
from sqlalchemy import Column
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import registry
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import declared_attr
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import Integer
from sqlalchemy.sql.sqltypes import String
reg: registry = registry()

@reg.mapped
class User:
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    name3 = Column(String(50))
    addresses: List['Address'] = relationship('Address')

@reg.mapped
class SubUser(User):
    __tablename__ = 'subuser'
    id: int = Column(ForeignKey('user.id'), primary_key=True)

    @declared_attr
    def name(cls) -> Column[String]:
        if False:
            print('Hello World!')
        return Column(String(50))

    @declared_attr
    def name2(cls) -> Mapped[Optional[str]]:
        if False:
            print('Hello World!')
        return Column(String(50))

    @declared_attr
    def name3(cls) -> Mapped[str]:
        if False:
            for i in range(10):
                print('nop')
        return Column(String(50))
    subname = Column(String)

@reg.mapped
class Address:
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)
    user_id: int = Column(ForeignKey('user.id'))
    email = Column(String(50))
    user = relationship(User, uselist=False)
s1 = SubUser()
x1: str = s1.name
x2: str = s1.name2
x3: str = s1.name3
u1 = User()
x4: str = u1.name3