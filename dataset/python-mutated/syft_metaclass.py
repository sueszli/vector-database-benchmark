import inspect
import threading
from typing import Any
from typing import Dict
from typing import Generator
from typing import Tuple
from typing import Type
from pydantic.fields import UndefinedType
from pydantic.main import BaseModel
from pydantic.main import ModelField
from pydantic.main import ModelMetaclass
from ..serde.recursive_primitives import recursive_serde_register_type
from ..serde.serializable import serializable
TupleGenerator = Generator[Tuple[str, Any], None, None]

@serializable()
class Empty:
    pass

class PartialModelMetaclass(ModelMetaclass):

    def __new__(meta: Type['PartialModelMetaclass'], *args: Any, **kwargs: Any) -> 'PartialModelMetaclass':
        if False:
            while True:
                i = 10
        cls = super().__new__(meta, *args, *kwargs)
        cls_init = cls.__init__
        init_lock = threading.Lock()

        def __init__(self: BaseModel, *args: Any, **kwargs: Any) -> None:
            if False:
                i = 10
                return i + 15
            with init_lock:
                fields = self.__class__.__fields__
                fields_map: Dict[ModelField, Tuple[Any, bool]] = {}

                def optionalize(fields: Dict[str, ModelField], *, restore: bool=False) -> None:
                    if False:
                        while True:
                            i = 10
                    for (_, field) in fields.items():
                        if not restore:
                            if isinstance(field.required, UndefinedType):
                                raise Exception(f'{field.name} is a required field.')
                            fields_map[field] = (field.type_, field.required)
                            if field.allow_none and field.name in kwargs:
                                field.required = True
                            else:
                                field.required = False
                            if inspect.isclass(field.type_) and issubclass(field.type_, BaseModel):
                                field.populate_validators()
                                if field.sub_fields is not None:
                                    for sub_field in field.sub_fields:
                                        sub_field.type_ = field.type_
                                        sub_field.populate_validators()
                                optionalize(field.type_.__fields__)
                        else:
                            (field.type_, field.required) = fields_map[field]
                            if field.sub_fields is not None:
                                for sub_field in field.sub_fields:
                                    sub_field.type_ = field.type_
                optionalize(fields)
                for (kwarg, value) in kwargs.items():
                    if value.__class__.__class__ is PartialModelMetaclass:
                        kwargs[kwarg] = value.dict()
                    elif isinstance(value, (tuple, list)):
                        kwargs[kwarg] = value.__class__((v.dict() if v.__class__.__class__ is PartialModelMetaclass else v for v in value))
                cls_init(self, *args, **kwargs)
                optionalize(fields, restore=True)
        cls.__init__ = __init__

        def iter_exclude_empty(self) -> TupleGenerator:
            if False:
                print('Hello World!')
            for (key, value) in self.__dict__.items():
                if value is not Empty:
                    yield (key, value)
        cls.__iter__ = iter_exclude_empty
        return cls
recursive_serde_register_type(PartialModelMetaclass)