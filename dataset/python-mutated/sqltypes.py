import uuid
from typing import Any, Optional, cast
from sqlalchemy import CHAR, types
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.type_api import TypeEngine

class AutoString(types.TypeDecorator):
    impl = types.String
    cache_ok = True
    mysql_default_length = 255

    def load_dialect_impl(self, dialect: Dialect) -> 'types.TypeEngine[Any]':
        if False:
            return 10
        impl = cast(types.String, self.impl)
        if impl.length is None and dialect.name == 'mysql':
            return dialect.type_descriptor(types.String(self.mysql_default_length))
        return super().load_dialect_impl(dialect)

class GUID(types.TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        if False:
            while True:
                i = 10
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        elif not isinstance(value, uuid.UUID):
            return uuid.UUID(value).hex
        else:
            return value.hex

    def process_result_value(self, value: Any, dialect: Dialect) -> Optional[uuid.UUID]:
        if False:
            while True:
                i = 10
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return cast(uuid.UUID, value)