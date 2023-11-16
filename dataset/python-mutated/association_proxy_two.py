from __future__ import annotations
from typing import Final
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.associationproxy import AssociationProxy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'user'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    kw: Mapped[list[Keyword]] = relationship(secondary=lambda : user_keyword_table)

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        self.name = name
    keywords: AssociationProxy[list[str]] = association_proxy('kw', 'keyword')

class Keyword(Base):
    __tablename__ = 'keyword'
    id: Mapped[int] = mapped_column(primary_key=True)
    keyword: Mapped[str] = mapped_column(String(64))

    def __init__(self, keyword: str):
        if False:
            for i in range(10):
                print('nop')
        self.keyword = keyword
user_keyword_table: Final[Table] = Table('user_keyword', Base.metadata, Column('user_id', Integer, ForeignKey('user.id'), primary_key=True), Column('keyword_id', Integer, ForeignKey('keyword.id'), primary_key=True))
user = User('jek')
reveal_type(user.kw)
user.kw.append(Keyword('cheese-inspector'))
user.keywords.append('cheese-inspector')
reveal_type(user.keywords)
user.keywords.append('snack ninja')