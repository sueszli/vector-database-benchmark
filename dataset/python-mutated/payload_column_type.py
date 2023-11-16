import json
from typing import Any, Generic, Type, TypeVar
import sqlalchemy.dialects.postgresql as pg
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, parse_obj_as, validator
from pydantic.main import ModelMetaclass
from sqlalchemy.types import TypeDecorator
payload_type_registry = {}
P = TypeVar('P', bound=BaseModel)

def payload_type(cls: Type[P]) -> Type[P]:
    if False:
        return 10
    payload_type_registry[cls.__name__] = cls
    return cls

class PayloadContainer(BaseModel):
    payload_type: str = ''
    payload: BaseModel = None

    def __init__(self, **v):
        if False:
            print('Hello World!')
        p = v['payload']
        if isinstance(p, dict):
            t = v['payload_type']
            if t not in payload_type_registry:
                raise RuntimeError(f"Payload type '{t}' not registered")
            cls = payload_type_registry[t]
            v['payload'] = cls(**p)
        super().__init__(**v)

    @validator('payload', pre=True)
    def check_payload(cls, v: BaseModel, values: dict[str, Any]) -> BaseModel:
        if False:
            print('Hello World!')
        values['payload_type'] = type(v).__name__
        return v

    class Config:
        orm_mode = True
T = TypeVar('T')

def payload_column_type(pydantic_type):
    if False:
        while True:
            i = 10

    class PayloadJSONBType(TypeDecorator, Generic[T]):
        impl = pg.JSONB()
        cache_ok = True

        def __init__(self, json_encoder=json):
            if False:
                for i in range(10):
                    print('nop')
            self.json_encoder = json_encoder
            super().__init__()

        def bind_processor(self, dialect):
            if False:
                i = 10
                return i + 15
            impl_processor = self.impl.bind_processor(dialect)
            dumps = self.json_encoder.dumps

            def process(value: T):
                if False:
                    i = 10
                    return i + 15
                if value is not None:
                    if isinstance(pydantic_type, ModelMetaclass):
                        value_to_dump = pydantic_type.from_orm(value)
                    else:
                        value_to_dump = value
                    value = jsonable_encoder(value_to_dump)
                if impl_processor:
                    return impl_processor(value)
                else:
                    return dumps(jsonable_encoder(value_to_dump))
            return process

        def result_processor(self, dialect, coltype) -> T:
            if False:
                print('Hello World!')
            impl_processor = self.impl.result_processor(dialect, coltype)

            def process(value):
                if False:
                    return 10
                if impl_processor:
                    value = impl_processor(value)
                if value is None:
                    return None
                full_obj = parse_obj_as(pydantic_type, value)
                return full_obj
            return process

        def compare_values(self, x, y):
            if False:
                i = 10
                return i + 15
            return x == y
    return PayloadJSONBType