from typing import List
from typing import Optional
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    addresses: List['Address'] = relationship('Address', back_populates='user')

    @property
    def some_property(self) -> List[Optional[int]]:
        if False:
            while True:
                i = 10
        return [i.id for i in self.addresses]

class Address(Base):
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)
    user_id: int = Column(ForeignKey('user.id'))
    user: 'User' = relationship('User', back_populates='addresses')

    @property
    def some_other_property(self) -> Optional[str]:
        if False:
            return 10
        return self.user.name
u1 = User(addresses=[Address()])
[x for x in u1.addresses]
stmt = select(User).where(User.addresses.any(id=5))