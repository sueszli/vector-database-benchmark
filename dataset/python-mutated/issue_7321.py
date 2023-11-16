from typing import Any
from typing import Dict
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import declared_attr
Base = declarative_base()

class Foo(Base):

    @declared_attr
    def __tablename__(cls) -> str:
        if False:
            print('Hello World!')
        return 'name'

    @declared_attr
    def __mapper_args__(cls) -> Dict[Any, Any]:
        if False:
            i = 10
            return i + 15
        return {}

    @declared_attr
    def __table_args__(cls) -> Dict[Any, Any]:
        if False:
            while True:
                i = 10
        return {}