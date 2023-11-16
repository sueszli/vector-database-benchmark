from typing import Any
from typing import Dict
from typing import Type
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import declared_attr
Base = declarative_base()

class Foo(Base):

    @declared_attr
    def __tablename__(cls: Type['Foo']) -> str:
        if False:
            print('Hello World!')
        return 'name'

    @declared_attr
    def __mapper_args__(cls: Type['Foo']) -> Dict[Any, Any]:
        if False:
            while True:
                i = 10
        return {}

    @classmethod
    @declared_attr
    def __table_args__(cls: Type['Foo']) -> Dict[Any, Any]:
        if False:
            return 10
        return {}