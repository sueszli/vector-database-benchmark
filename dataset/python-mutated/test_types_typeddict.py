"""
Tests for TypedDict
"""
import sys
import typing
from typing import Any, Dict, Generic, List, Optional, TypeVar
import pytest
import typing_extensions
from annotated_types import Lt
from pydantic_core import CoreSchema, core_schema
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, ConfigDict, Field, GenerateSchema, GetCoreSchemaHandler, PositiveInt, PydanticUserError, ValidationError
from pydantic._internal._decorators import get_attribute_from_bases
from pydantic.functional_serializers import field_serializer, model_serializer
from pydantic.functional_validators import field_validator, model_validator
from pydantic.type_adapter import TypeAdapter
from .conftest import Err

@pytest.fixture(name='TypedDictAll', params=[pytest.param(typing, id='typing.TypedDict'), pytest.param(typing_extensions, id='t_e.TypedDict')])
def fixture_typed_dict_all(request):
    if False:
        print('Hello World!')
    try:
        return request.param.TypedDict
    except AttributeError:
        pytest.skip(f'TypedDict is not available from {request.param}')

@pytest.fixture(name='TypedDict')
def fixture_typed_dict(TypedDictAll):
    if False:
        return 10

    class TestTypedDict(TypedDictAll):
        foo: str
    if sys.version_info < (3, 12) and TypedDictAll.__module__ == 'typing':
        pytest.skip('typing.TypedDict does not support all pydantic features in Python < 3.12')
    if hasattr(TestTypedDict, '__required_keys__'):
        return TypedDictAll
    else:
        pytest.skip('TypedDict does not include __required_keys__')

@pytest.fixture(name='req_no_req', params=[pytest.param(typing, id='typing.Required'), pytest.param(typing_extensions, id='t_e.Required')])
def fixture_req_no_req(request):
    if False:
        print('Hello World!')
    try:
        return (request.param.Required, request.param.NotRequired)
    except AttributeError:
        pytest.skip(f'Required and NotRequired are not available from {request.param}')

def test_typeddict_all(TypedDictAll):
    if False:
        while True:
            i = 10

    class MyDict(TypedDictAll):
        foo: str
    try:

        class M(BaseModel):
            d: MyDict
    except PydanticUserError as e:
        assert e.message == 'Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.'
    else:
        assert M(d=dict(foo='baz')).d == {'foo': 'baz'}

def test_typeddict_annotated_simple(TypedDict, req_no_req):
    if False:
        i = 10
        return i + 15
    (Required, NotRequired) = req_no_req

    class MyDict(TypedDict):
        foo: str
        bar: Annotated[int, Lt(10)]
        spam: NotRequired[float]

    class M(BaseModel):
        d: MyDict
    assert M(d=dict(foo='baz', bar='8')).d == {'foo': 'baz', 'bar': 8}
    assert M(d=dict(foo='baz', bar='8', spam='44.4')).d == {'foo': 'baz', 'bar': 8, 'spam': 44.4}
    with pytest.raises(ValidationError, match='d\\.bar\\s+Field required \\[type=missing,'):
        M(d=dict(foo='baz'))
    with pytest.raises(ValidationError, match='d\\.bar\\s+Input should be less than 10 \\[type=less_than,'):
        M(d=dict(foo='baz', bar='11'))

def test_typeddict_total_false(TypedDict, req_no_req):
    if False:
        print('Hello World!')
    (Required, NotRequired) = req_no_req

    class MyDict(TypedDict, total=False):
        foo: Required[str]
        bar: int

    class M(BaseModel):
        d: MyDict
    assert M(d=dict(foo='baz', bar='8')).d == {'foo': 'baz', 'bar': 8}
    assert M(d=dict(foo='baz')).d == {'foo': 'baz'}
    with pytest.raises(ValidationError, match='d\\.foo\\s+Field required \\[type=missing,'):
        M(d={})

def test_typeddict(TypedDict):
    if False:
        i = 10
        return i + 15

    class TD(TypedDict):
        a: int
        b: int
        c: int
        d: str

    class Model(BaseModel):
        td: TD
    m = Model(td={'a': '3', 'b': b'1', 'c': 4, 'd': 'qwe'})
    assert m.td == {'a': 3, 'b': 1, 'c': 4, 'd': 'qwe'}
    with pytest.raises(ValidationError) as exc_info:
        Model(td={'a': [1], 'b': 2, 'c': 3, 'd': 'qwe'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('td', 'a'), 'msg': 'Input should be a valid integer', 'input': [1]}]

def test_typeddict_non_total(TypedDict):
    if False:
        for i in range(10):
            print('nop')

    class FullMovie(TypedDict, total=True):
        name: str
        year: int

    class Model(BaseModel):
        movie: FullMovie
    with pytest.raises(ValidationError) as exc_info:
        Model(movie={'year': '2002'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing', 'loc': ('movie', 'name'), 'msg': 'Field required', 'input': {'year': '2002'}}]

    class PartialMovie(TypedDict, total=False):
        name: str
        year: int

    class Model(BaseModel):
        movie: PartialMovie
    m = Model(movie={'year': '2002'})
    assert m.movie == {'year': 2002}

def test_partial_new_typeddict(TypedDict):
    if False:
        for i in range(10):
            print('nop')

    class OptionalUser(TypedDict, total=False):
        name: str

    class User(OptionalUser):
        id: int

    class Model(BaseModel):
        user: User
    assert Model(user={'id': 1, 'name': 'foobar'}).user == {'id': 1, 'name': 'foobar'}
    assert Model(user={'id': 1}).user == {'id': 1}

def test_typeddict_extra_default(TypedDict):
    if False:
        i = 10
        return i + 15

    class User(TypedDict):
        name: str
        age: int
    ta = TypeAdapter(User)
    assert ta.validate_python({'name': 'pika', 'age': 7, 'rank': 1}) == {'name': 'pika', 'age': 7}

    class UserExtraAllow(User):
        __pydantic_config__ = ConfigDict(extra='allow')
    ta = TypeAdapter(UserExtraAllow)
    assert ta.validate_python({'name': 'pika', 'age': 7, 'rank': 1}) == {'name': 'pika', 'age': 7, 'rank': 1}

    class UserExtraForbid(User):
        __pydantic_config__ = ConfigDict(extra='forbid')
    ta = TypeAdapter(UserExtraForbid)
    with pytest.raises(ValidationError) as exc_info:
        ta.validate_python({'name': 'pika', 'age': 7, 'rank': 1})
    assert exc_info.value.errors(include_url=False) == [{'type': 'extra_forbidden', 'loc': ('rank',), 'msg': 'Extra inputs are not permitted', 'input': 1}]

def test_typeddict_schema(TypedDict):
    if False:
        i = 10
        return i + 15

    class Data(BaseModel):
        a: int

    class DataTD(TypedDict):
        a: int

    class CustomTD(TypedDict):
        b: int

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            if False:
                return 10
            schema = handler(source_type)
            schema = handler.resolve_ref_schema(schema)
            assert schema['type'] == 'typed-dict'
            b = schema['fields']['b']['schema']
            assert b['type'] == 'int'
            b['gt'] = 0
            return schema

    class Model(BaseModel):
        data: Data
        data_td: DataTD
        custom_td: CustomTD
    assert Model.model_json_schema(mode='validation') == {'type': 'object', 'properties': {'data': {'$ref': '#/$defs/Data'}, 'data_td': {'$ref': '#/$defs/DataTD'}, 'custom_td': {'$ref': '#/$defs/CustomTD'}}, 'required': ['data', 'data_td', 'custom_td'], 'title': 'Model', '$defs': {'DataTD': {'type': 'object', 'properties': {'a': {'type': 'integer', 'title': 'A'}}, 'required': ['a'], 'title': 'DataTD'}, 'CustomTD': {'type': 'object', 'properties': {'b': {'type': 'integer', 'exclusiveMinimum': 0, 'title': 'B'}}, 'required': ['b'], 'title': 'CustomTD'}, 'Data': {'type': 'object', 'properties': {'a': {'type': 'integer', 'title': 'A'}}, 'required': ['a'], 'title': 'Data'}}}
    assert Model.model_json_schema(mode='serialization') == {'type': 'object', 'properties': {'data': {'$ref': '#/$defs/Data'}, 'data_td': {'$ref': '#/$defs/DataTD'}, 'custom_td': {'$ref': '#/$defs/CustomTD'}}, 'required': ['data', 'data_td', 'custom_td'], 'title': 'Model', '$defs': {'DataTD': {'type': 'object', 'properties': {'a': {'type': 'integer', 'title': 'A'}}, 'required': ['a'], 'title': 'DataTD'}, 'CustomTD': {'type': 'object', 'properties': {'b': {'type': 'integer', 'exclusiveMinimum': 0, 'title': 'B'}}, 'required': ['b'], 'title': 'CustomTD'}, 'Data': {'type': 'object', 'properties': {'a': {'type': 'integer', 'title': 'A'}}, 'required': ['a'], 'title': 'Data'}}}

def test_typeddict_postponed_annotation(TypedDict):
    if False:
        print('Hello World!')

    class DataTD(TypedDict):
        v: 'PositiveInt'

    class Model(BaseModel):
        t: DataTD
    with pytest.raises(ValidationError):
        Model.model_validate({'t': {'v': -1}})

def test_typeddict_required(TypedDict, req_no_req):
    if False:
        return 10
    (Required, _) = req_no_req

    class DataTD(TypedDict, total=False):
        a: int
        b: Required[str]

    class Model(BaseModel):
        t: DataTD
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'t': {'$ref': '#/$defs/DataTD'}}, 'required': ['t'], '$defs': {'DataTD': {'title': 'DataTD', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'string'}}, 'required': ['b']}}}

def test_typeddict_from_attributes():
    if False:
        while True:
            i = 10

    class UserCls:

        def __init__(self, name: str, age: int):
            if False:
                print('Hello World!')
            self.name = name
            self.age = age

    class User(TypedDict):
        name: str
        age: int

    class FromAttributesCls:

        def __init__(self, u: User):
            if False:
                while True:
                    i = 10
            self.u = u

    class Model(BaseModel):
        u: Annotated[User, Field(strict=False)]

    class FromAttributesModel(BaseModel, from_attributes=True):
        u: Annotated[User, Field(strict=False)]
    assert FromAttributesModel.model_validate(FromAttributesCls(u={'name': 'foo', 'age': 15}))
    with pytest.raises(ValidationError, match='Input should be a valid dictionary'):
        Model(u=UserCls('foo', 15))
    with pytest.raises(ValidationError, match='Input should be a valid dictionary'):
        FromAttributesModel(u=UserCls('foo', 15))

def test_typeddict_not_required_schema(TypedDict, req_no_req):
    if False:
        print('Hello World!')
    (Required, NotRequired) = req_no_req

    class DataTD(TypedDict, total=True):
        a: NotRequired[int]
        b: str

    class Model(BaseModel):
        t: DataTD
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'t': {'$ref': '#/$defs/DataTD'}}, 'required': ['t'], '$defs': {'DataTD': {'title': 'DataTD', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'string'}}, 'required': ['b']}}}

def test_typed_dict_inheritance_schema(TypedDict, req_no_req):
    if False:
        i = 10
        return i + 15
    (Required, NotRequired) = req_no_req

    class DataTDBase(TypedDict, total=True):
        a: NotRequired[int]
        b: str

    class DataTD(DataTDBase, total=False):
        c: Required[int]
        d: str

    class Model(BaseModel):
        t: DataTD
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'t': {'$ref': '#/$defs/DataTD'}}, 'required': ['t'], '$defs': {'DataTD': {'title': 'DataTD', 'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'string'}, 'c': {'title': 'C', 'type': 'integer'}, 'd': {'title': 'D', 'type': 'string'}}, 'required': ['b', 'c']}}}

def test_typeddict_annotated_nonoptional_schema(TypedDict):
    if False:
        print('Hello World!')

    class DataTD(TypedDict):
        a: Optional[int]
        b: Annotated[Optional[int], Field(42)]
        c: Annotated[Optional[int], Field(description='Test')]

    class Model(BaseModel):
        data_td: DataTD
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', 'properties': {'data_td': {'$ref': '#/$defs/DataTD'}}, 'required': ['data_td'], '$defs': {'DataTD': {'type': 'object', 'title': 'DataTD', 'properties': {'a': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'A'}, 'b': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'default': 42, 'title': 'B'}, 'c': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'description': 'Test', 'title': 'C'}}, 'required': ['a', 'c']}}}

@pytest.mark.parametrize('input_value,expected', [({'a': '1', 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}), ({'a': None, 'b': 2, 'c': 3}, {'a': None, 'b': 2, 'c': 3}), ({'a': None, 'c': 3}, {'a': None, 'b': 42, 'c': 3})], ids=repr)
def test_typeddict_annotated(TypedDict, input_value, expected):
    if False:
        i = 10
        return i + 15

    class DataTD(TypedDict):
        a: Optional[int]
        b: Annotated[Optional[int], Field(42)]
        c: Annotated[Optional[int], Field(description='Test', lt=4)]

    class Model(BaseModel):
        d: DataTD
    if isinstance(expected, Err):
        with pytest.raises(ValidationError, match=expected.message_escaped()):
            Model(d=input_value)
    else:
        assert Model(d=input_value).d == expected

def test_recursive_typeddict():
    if False:
        while True:
            i = 10
    from typing import Optional
    from typing_extensions import TypedDict
    from pydantic import BaseModel

    class RecursiveTypedDict(TypedDict):
        foo: Optional['RecursiveTypedDict']

    class RecursiveTypedDictModel(BaseModel):
        rec: RecursiveTypedDict
    assert RecursiveTypedDictModel(rec={'foo': {'foo': None}}).rec == {'foo': {'foo': None}}
    with pytest.raises(ValidationError) as exc_info:
        RecursiveTypedDictModel(rec={'foo': {'foo': {'foo': 1}}})
    assert exc_info.value.errors(include_url=False) == [{'input': 1, 'loc': ('rec', 'foo', 'foo', 'foo'), 'msg': 'Input should be a valid dictionary', 'type': 'dict_type'}]
T = TypeVar('T')

def test_generic_typeddict_in_concrete_model():
    if False:
        return 10
    T = TypeVar('T')

    class GenericTypedDict(typing_extensions.TypedDict, Generic[T]):
        x: T

    class Model(BaseModel):
        y: GenericTypedDict[int]
    Model(y={'x': 1})
    with pytest.raises(ValidationError) as exc_info:
        Model(y={'x': 'a'})
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('y', 'x'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_generic_typeddict_in_generic_model():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    class GenericTypedDict(typing_extensions.TypedDict, Generic[T]):
        x: T

    class Model(BaseModel, Generic[T]):
        y: GenericTypedDict[T]
    Model[int](y={'x': 1})
    with pytest.raises(ValidationError) as exc_info:
        Model[int](y={'x': 'a'})
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('y', 'x'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_recursive_generic_typeddict_in_module(create_module):
    if False:
        i = 10
        return i + 15

    @create_module
    def module():
        if False:
            print('Hello World!')
        from typing import Generic, List, Optional, TypeVar
        from typing_extensions import TypedDict
        from pydantic import BaseModel
        T = TypeVar('T')

        class RecursiveGenTypedDictModel(BaseModel, Generic[T]):
            rec: 'RecursiveGenTypedDict[T]'

        class RecursiveGenTypedDict(TypedDict, Generic[T]):
            foo: Optional['RecursiveGenTypedDict[T]']
            ls: List[T]
    int_data: module.RecursiveGenTypedDict[int] = {'foo': {'foo': None, 'ls': [1]}, 'ls': [1]}
    assert module.RecursiveGenTypedDictModel[int](rec=int_data).rec == int_data
    str_data: module.RecursiveGenTypedDict[str] = {'foo': {'foo': None, 'ls': ['a']}, 'ls': ['a']}
    with pytest.raises(ValidationError) as exc_info:
        module.RecursiveGenTypedDictModel[int](rec=str_data)
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('rec', 'foo', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}, {'input': 'a', 'loc': ('rec', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_recursive_generic_typeddict_in_function_1():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    class RecursiveGenTypedDict(TypedDict, Generic[T]):
        foo: Optional['RecursiveGenTypedDict[T]']
        ls: List[T]

    class RecursiveGenTypedDictModel(BaseModel, Generic[T]):
        rec: 'RecursiveGenTypedDict[T]'
    int_data: RecursiveGenTypedDict[int] = {'foo': {'foo': None, 'ls': [1]}, 'ls': [1]}
    assert RecursiveGenTypedDictModel[int](rec=int_data).rec == int_data
    str_data: RecursiveGenTypedDict[str] = {'foo': {'foo': None, 'ls': ['a']}, 'ls': ['a']}
    with pytest.raises(ValidationError) as exc_info:
        RecursiveGenTypedDictModel[int](rec=str_data)
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('rec', 'foo', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}, {'input': 'a', 'loc': ('rec', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_recursive_generic_typeddict_in_function_2():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    class RecursiveGenTypedDictModel(BaseModel, Generic[T]):
        rec: 'RecursiveGenTypedDict[T]'

    class RecursiveGenTypedDict(TypedDict, Generic[T]):
        foo: Optional['RecursiveGenTypedDict[T]']
        ls: List[T]
    int_data: RecursiveGenTypedDict[int] = {'foo': {'foo': None, 'ls': [1]}, 'ls': [1]}
    assert RecursiveGenTypedDictModel[int](rec=int_data).rec == int_data
    str_data: RecursiveGenTypedDict[str] = {'foo': {'foo': None, 'ls': ['a']}, 'ls': ['a']}
    with pytest.raises(ValidationError) as exc_info:
        RecursiveGenTypedDictModel[int](rec=str_data)
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('rec', 'foo', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}, {'input': 'a', 'loc': ('rec', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_recursive_generic_typeddict_in_function_3():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    class RecursiveGenTypedDictModel(BaseModel, Generic[T]):
        rec: 'RecursiveGenTypedDict[T]'
    IntModel = RecursiveGenTypedDictModel[int]

    class RecursiveGenTypedDict(TypedDict, Generic[T]):
        foo: Optional['RecursiveGenTypedDict[T]']
        ls: List[T]
    int_data: RecursiveGenTypedDict[int] = {'foo': {'foo': None, 'ls': [1]}, 'ls': [1]}
    assert IntModel(rec=int_data).rec == int_data
    str_data: RecursiveGenTypedDict[str] = {'foo': {'foo': None, 'ls': ['a']}, 'ls': ['a']}
    with pytest.raises(ValidationError) as exc_info:
        IntModel(rec=str_data)
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('rec', 'foo', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}, {'input': 'a', 'loc': ('rec', 'ls', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_typeddict_alias_generator(TypedDict):
    if False:
        i = 10
        return i + 15

    def alias_generator(name: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'alias_' + name

    class MyDict(TypedDict):
        __pydantic_config__ = ConfigDict(alias_generator=alias_generator, extra='forbid')
        foo: str

    class Model(BaseModel):
        d: MyDict
    ta = TypeAdapter(MyDict)
    model = ta.validate_python({'alias_foo': 'bar'})
    assert model['foo'] == 'bar'
    with pytest.raises(ValidationError) as exc_info:
        ta.validate_python({'foo': 'bar'})
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing', 'loc': ('alias_foo',), 'msg': 'Field required', 'input': {'foo': 'bar'}}, {'input': 'bar', 'loc': ('foo',), 'msg': 'Extra inputs are not permitted', 'type': 'extra_forbidden'}]

def test_typeddict_inheritence(TypedDict: Any) -> None:
    if False:
        return 10

    class Parent(TypedDict):
        x: int

    class Child(Parent):
        y: float
    ta = TypeAdapter(Child)
    assert ta.validate_python({'x': '1', 'y': '1.0'}) == {'x': 1, 'y': 1.0}

def test_typeddict_field_validator(TypedDict: Any) -> None:
    if False:
        for i in range(10):
            print('nop')

    class Parent(TypedDict):
        a: List[str]

        @field_validator('a')
        @classmethod
        def parent_val_before(cls, v: List[str]):
            if False:
                i = 10
                return i + 15
            v.append('parent before')
            return v

        @field_validator('a')
        @classmethod
        def val(cls, v: List[str]):
            if False:
                return 10
            v.append('parent')
            return v

        @field_validator('a')
        @classmethod
        def parent_val_after(cls, v: List[str]):
            if False:
                return 10
            v.append('parent after')
            return v

    class Child(Parent):

        @field_validator('a')
        @classmethod
        def child_val_before(cls, v: List[str]):
            if False:
                while True:
                    i = 10
            v.append('child before')
            return v

        @field_validator('a')
        @classmethod
        def val(cls, v: List[str]):
            if False:
                return 10
            v.append('child')
            return v

        @field_validator('a')
        @classmethod
        def child_val_after(cls, v: List[str]):
            if False:
                print('Hello World!')
            v.append('child after')
            return v
    parent_ta = TypeAdapter(Parent)
    child_ta = TypeAdapter(Child)
    assert parent_ta.validate_python({'a': []})['a'] == ['parent before', 'parent', 'parent after']
    assert child_ta.validate_python({'a': []})['a'] == ['parent before', 'child', 'parent after', 'child before', 'child after']

def test_typeddict_model_validator(TypedDict) -> None:
    if False:
        print('Hello World!')

    class Model(TypedDict):
        x: int
        y: float

        @model_validator(mode='before')
        @classmethod
        def val_model_before(cls, value: Dict[str, Any]) -> Dict[str, Any]:
            if False:
                i = 10
                return i + 15
            return dict(x=value['x'] + 1, y=value['y'] + 2)

        @model_validator(mode='after')
        def val_model_after(self) -> 'Model':
            if False:
                while True:
                    i = 10
            return Model(x=self['x'] * 2, y=self['y'] * 3)
    ta = TypeAdapter(Model)
    assert ta.validate_python({'x': 1, 'y': 2.5}) == {'x': 4, 'y': 13.5}

def test_typeddict_field_serializer(TypedDict: Any) -> None:
    if False:
        return 10

    class Parent(TypedDict):
        a: List[str]

        @field_serializer('a')
        @classmethod
        def ser(cls, v: List[str]):
            if False:
                i = 10
                return i + 15
            v.append('parent')
            return v

    class Child(Parent):

        @field_serializer('a')
        @classmethod
        def ser(cls, v: List[str]):
            if False:
                while True:
                    i = 10
            v.append('child')
            return v
    parent_ta = TypeAdapter(Parent)
    child_ta = TypeAdapter(Child)
    assert parent_ta.dump_python(Parent({'a': []}))['a'] == ['parent']
    assert child_ta.dump_python(Child({'a': []}))['a'] == ['child']

def test_typeddict_model_serializer(TypedDict) -> None:
    if False:
        for i in range(10):
            print('nop')

    class Model(TypedDict):
        x: int
        y: float

        @model_serializer(mode='plain')
        def ser_model(self) -> Dict[str, Any]:
            if False:
                i = 10
                return i + 15
            return {'x': self['x'] * 2, 'y': self['y'] * 3}
    ta = TypeAdapter(Model)
    assert ta.dump_python(Model({'x': 1, 'y': 2.5})) == {'x': 2, 'y': 7.5}

def test_model_config() -> None:
    if False:
        return 10

    class Model(TypedDict):
        x: str
        __pydantic_config__ = ConfigDict(str_to_lower=True)
    ta = TypeAdapter(Model)
    assert ta.validate_python({'x': 'ABC'}) == {'x': 'abc'}

def test_model_config_inherited() -> None:
    if False:
        print('Hello World!')

    class Base(TypedDict):
        __pydantic_config__ = ConfigDict(str_to_lower=True)

    class Model(Base):
        x: str
    ta = TypeAdapter(Model)
    assert ta.validate_python({'x': 'ABC'}) == {'x': 'abc'}

def test_schema_generator() -> None:
    if False:
        for i in range(10):
            print('nop')

    class LaxStrGenerator(GenerateSchema):

        def str_schema(self) -> CoreSchema:
            if False:
                i = 10
                return i + 15
            return core_schema.no_info_plain_validator_function(str)

    class Model(TypedDict):
        x: str
        __pydantic_config__ = ConfigDict(schema_generator=LaxStrGenerator)
    ta = TypeAdapter(Model)
    assert ta.validate_python(dict(x=1))['x'] == '1'

def test_grandparent_config():
    if False:
        for i in range(10):
            print('nop')

    class MyTypedDict(TypedDict):
        __pydantic_config__ = ConfigDict(str_to_lower=True)
        x: str

    class MyMiddleTypedDict(MyTypedDict):
        y: str

    class MySubTypedDict(MyMiddleTypedDict):
        z: str
    validated_data = TypeAdapter(MySubTypedDict).validate_python({'x': 'ABC', 'y': 'DEF', 'z': 'GHI'})
    assert validated_data == {'x': 'abc', 'y': 'def', 'z': 'ghi'}

def test_typeddict_mro():
    if False:
        while True:
            i = 10

    class A(TypedDict):
        x = 1

    class B(A):
        x = 2

    class C(B):
        pass
    assert get_attribute_from_bases(C, 'x') == 2