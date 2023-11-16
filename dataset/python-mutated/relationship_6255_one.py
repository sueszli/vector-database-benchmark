from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = mapped_column(Integer, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    addresses: Mapped[List['Address']] = relationship('Address', back_populates='user')

    @property
    def some_property(self) -> List[Optional[int]]:
        if False:
            i = 10
            return i + 15
        return [i.id for i in self.addresses]

class Address(Base):
    __tablename__ = 'address'
    id = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'))
    user: Mapped['User'] = relationship('User', back_populates='addresses')

    @property
    def some_other_property(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self.user.name
u1 = User(addresses=[Address()])
[x for x in u1.addresses]
stmt = select(User).where(User.addresses.any(id=5))