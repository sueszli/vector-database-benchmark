import typing
from sqlalchemy import Integer
from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

class Base(DeclarativeBase):
    pass

class HasRelatedDataMixin:

    @declared_attr
    def related_data(cls) -> Mapped[str]:
        if False:
            for i in range(10):
                print('nop')
        return mapped_column(Text(), deferred=True)

class User(HasRelatedDataMixin, Base):

    @declared_attr.directive
    def __tablename__(cls) -> str:
        if False:
            i = 10
            return i + 15
        return 'user'

    @declared_attr.directive
    def __mapper_args__(cls) -> typing.Dict[str, typing.Any]:
        if False:
            for i in range(10):
                print('nop')
        return {}
    id = mapped_column(Integer, primary_key=True)

class Foo(Base):
    __tablename__ = 'foo'
    id = mapped_column(Integer, primary_key=True)
u1 = User()
if typing.TYPE_CHECKING:
    reveal_type(User.__tablename__)
    reveal_type(Foo.__tablename__)
    reveal_type(u1.related_data)
    reveal_type(User.related_data)