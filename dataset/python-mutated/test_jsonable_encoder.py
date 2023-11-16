from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Optional
import pytest
from fastapi._compat import PYDANTIC_V2
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, ValidationError
from .utils import needs_pydanticv1, needs_pydanticv2

class Person:

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        self.name = name

class Pet:

    def __init__(self, owner: Person, name: str):
        if False:
            while True:
                i = 10
        self.owner = owner
        self.name = name

@dataclass
class Item:
    name: str
    count: int

class DictablePerson(Person):

    def __iter__(self):
        if False:
            while True:
                i = 10
        return ((k, v) for (k, v) in self.__dict__.items())

class DictablePet(Pet):

    def __iter__(self):
        if False:
            return 10
        return ((k, v) for (k, v) in self.__dict__.items())

class Unserializable:

    def __iter__(self):
        if False:
            return 10
        raise NotImplementedError()

    @property
    def __dict__(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class RoleEnum(Enum):
    admin = 'admin'
    normal = 'normal'

class ModelWithConfig(BaseModel):
    role: Optional[RoleEnum] = None
    if PYDANTIC_V2:
        model_config = {'use_enum_values': True}
    else:

        class Config:
            use_enum_values = True

class ModelWithAlias(BaseModel):
    foo: str = Field(alias='Foo')

class ModelWithDefault(BaseModel):
    foo: str = ...
    bar: str = 'bar'
    bla: str = 'bla'

def test_encode_dict():
    if False:
        return 10
    pet = {'name': 'Firulais', 'owner': {'name': 'Foo'}}
    assert jsonable_encoder(pet) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}
    assert jsonable_encoder(pet, include={'name'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, exclude={'owner'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, include={}) == {}
    assert jsonable_encoder(pet, exclude={}) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}

def test_encode_class():
    if False:
        print('Hello World!')
    person = Person(name='Foo')
    pet = Pet(owner=person, name='Firulais')
    assert jsonable_encoder(pet) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}
    assert jsonable_encoder(pet, include={'name'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, exclude={'owner'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, include={}) == {}
    assert jsonable_encoder(pet, exclude={}) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}

def test_encode_dictable():
    if False:
        for i in range(10):
            print('nop')
    person = DictablePerson(name='Foo')
    pet = DictablePet(owner=person, name='Firulais')
    assert jsonable_encoder(pet) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}
    assert jsonable_encoder(pet, include={'name'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, exclude={'owner'}) == {'name': 'Firulais'}
    assert jsonable_encoder(pet, include={}) == {}
    assert jsonable_encoder(pet, exclude={}) == {'name': 'Firulais', 'owner': {'name': 'Foo'}}

def test_encode_dataclass():
    if False:
        i = 10
        return i + 15
    item = Item(name='foo', count=100)
    assert jsonable_encoder(item) == {'name': 'foo', 'count': 100}
    assert jsonable_encoder(item, include={'name'}) == {'name': 'foo'}
    assert jsonable_encoder(item, exclude={'count'}) == {'name': 'foo'}
    assert jsonable_encoder(item, include={}) == {}
    assert jsonable_encoder(item, exclude={}) == {'name': 'foo', 'count': 100}

def test_encode_unsupported():
    if False:
        print('Hello World!')
    unserializable = Unserializable()
    with pytest.raises(ValueError):
        jsonable_encoder(unserializable)

@needs_pydanticv2
def test_encode_custom_json_encoders_model_pydanticv2():
    if False:
        return 10
    from pydantic import field_serializer

    class ModelWithCustomEncoder(BaseModel):
        dt_field: datetime

        @field_serializer('dt_field')
        def serialize_dt_field(self, dt):
            if False:
                for i in range(10):
                    print('nop')
            return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat()

    class ModelWithCustomEncoderSubclass(ModelWithCustomEncoder):
        pass
    model = ModelWithCustomEncoder(dt_field=datetime(2019, 1, 1, 8))
    assert jsonable_encoder(model) == {'dt_field': '2019-01-01T08:00:00+00:00'}
    subclass_model = ModelWithCustomEncoderSubclass(dt_field=datetime(2019, 1, 1, 8))
    assert jsonable_encoder(subclass_model) == {'dt_field': '2019-01-01T08:00:00+00:00'}

@needs_pydanticv1
def test_encode_custom_json_encoders_model_pydanticv1():
    if False:
        for i in range(10):
            print('nop')

    class ModelWithCustomEncoder(BaseModel):
        dt_field: datetime

        class Config:
            json_encoders = {datetime: lambda dt: dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat()}

    class ModelWithCustomEncoderSubclass(ModelWithCustomEncoder):

        class Config:
            pass
    model = ModelWithCustomEncoder(dt_field=datetime(2019, 1, 1, 8))
    assert jsonable_encoder(model) == {'dt_field': '2019-01-01T08:00:00+00:00'}
    subclass_model = ModelWithCustomEncoderSubclass(dt_field=datetime(2019, 1, 1, 8))
    assert jsonable_encoder(subclass_model) == {'dt_field': '2019-01-01T08:00:00+00:00'}

def test_encode_model_with_config():
    if False:
        for i in range(10):
            print('nop')
    model = ModelWithConfig(role=RoleEnum.admin)
    assert jsonable_encoder(model) == {'role': 'admin'}

def test_encode_model_with_alias_raises():
    if False:
        print('Hello World!')
    with pytest.raises(ValidationError):
        ModelWithAlias(foo='Bar')

def test_encode_model_with_alias():
    if False:
        return 10
    model = ModelWithAlias(Foo='Bar')
    assert jsonable_encoder(model) == {'Foo': 'Bar'}

def test_encode_model_with_default():
    if False:
        return 10
    model = ModelWithDefault(foo='foo', bar='bar')
    assert jsonable_encoder(model) == {'foo': 'foo', 'bar': 'bar', 'bla': 'bla'}
    assert jsonable_encoder(model, exclude_unset=True) == {'foo': 'foo', 'bar': 'bar'}
    assert jsonable_encoder(model, exclude_defaults=True) == {'foo': 'foo'}
    assert jsonable_encoder(model, exclude_unset=True, exclude_defaults=True) == {'foo': 'foo'}
    assert jsonable_encoder(model, include={'foo'}) == {'foo': 'foo'}
    assert jsonable_encoder(model, exclude={'bla'}) == {'foo': 'foo', 'bar': 'bar'}
    assert jsonable_encoder(model, include={}) == {}
    assert jsonable_encoder(model, exclude={}) == {'foo': 'foo', 'bar': 'bar', 'bla': 'bla'}

@needs_pydanticv1
def test_custom_encoders():
    if False:
        return 10

    class safe_datetime(datetime):
        pass

    class MyModel(BaseModel):
        dt_field: safe_datetime
    instance = MyModel(dt_field=safe_datetime.now())
    encoded_instance = jsonable_encoder(instance, custom_encoder={safe_datetime: lambda o: o.isoformat()})
    assert encoded_instance['dt_field'] == instance.dt_field.isoformat()

def test_custom_enum_encoders():
    if False:
        return 10

    def custom_enum_encoder(v: Enum):
        if False:
            while True:
                i = 10
        return v.value.lower()

    class MyEnum(Enum):
        ENUM_VAL_1 = 'ENUM_VAL_1'
    instance = MyEnum.ENUM_VAL_1
    encoded_instance = jsonable_encoder(instance, custom_encoder={MyEnum: custom_enum_encoder})
    assert encoded_instance == custom_enum_encoder(instance)

def test_encode_model_with_pure_path():
    if False:
        print('Hello World!')

    class ModelWithPath(BaseModel):
        path: PurePath
        if PYDANTIC_V2:
            model_config = {'arbitrary_types_allowed': True}
        else:

            class Config:
                arbitrary_types_allowed = True
    test_path = PurePath('/foo', 'bar')
    obj = ModelWithPath(path=test_path)
    assert jsonable_encoder(obj) == {'path': str(test_path)}

def test_encode_model_with_pure_posix_path():
    if False:
        while True:
            i = 10

    class ModelWithPath(BaseModel):
        path: PurePosixPath
        if PYDANTIC_V2:
            model_config = {'arbitrary_types_allowed': True}
        else:

            class Config:
                arbitrary_types_allowed = True
    obj = ModelWithPath(path=PurePosixPath('/foo', 'bar'))
    assert jsonable_encoder(obj) == {'path': '/foo/bar'}

def test_encode_model_with_pure_windows_path():
    if False:
        for i in range(10):
            print('nop')

    class ModelWithPath(BaseModel):
        path: PureWindowsPath
        if PYDANTIC_V2:
            model_config = {'arbitrary_types_allowed': True}
        else:

            class Config:
                arbitrary_types_allowed = True
    obj = ModelWithPath(path=PureWindowsPath('/foo', 'bar'))
    assert jsonable_encoder(obj) == {'path': '\\foo\\bar'}

@needs_pydanticv1
def test_encode_root():
    if False:
        return 10

    class ModelWithRoot(BaseModel):
        __root__: str
    model = ModelWithRoot(__root__='Foo')
    assert jsonable_encoder(model) == 'Foo'

@needs_pydanticv2
def test_decimal_encoder_float():
    if False:
        i = 10
        return i + 15
    data = {'value': Decimal(1.23)}
    assert jsonable_encoder(data) == {'value': 1.23}

@needs_pydanticv2
def test_decimal_encoder_int():
    if False:
        return 10
    data = {'value': Decimal(2)}
    assert jsonable_encoder(data) == {'value': 2}

def test_encode_deque_encodes_child_models():
    if False:
        i = 10
        return i + 15

    class Model(BaseModel):
        test: str
    dq = deque([Model(test='test')])
    assert jsonable_encoder(dq)[0]['test'] == 'test'