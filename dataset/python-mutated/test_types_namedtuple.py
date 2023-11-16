from collections import namedtuple
from typing import Generic, NamedTuple, Optional, Tuple, TypeVar
import pytest
from typing_extensions import NamedTuple as TypingExtensionsNamedTuple
from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, ValidationError
from pydantic.errors import PydanticSchemaGenerationError

def test_namedtuple_simple():
    if False:
        return 10
    Position = namedtuple('Pos', 'x y')

    class Model(BaseModel):
        pos: Position
    model = Model(pos=('1', 2))
    assert isinstance(model.pos, Position)
    assert model.pos.x == '1'
    assert model.pos == Position('1', 2)
    model = Model(pos={'x': '1', 'y': 2})
    assert model.pos == Position('1', 2)

def test_namedtuple():
    if False:
        while True:
            i = 10

    class Event(NamedTuple):
        a: int
        b: int
        c: int
        d: str

    class Model(BaseModel):
        event: Event
    model = Model(event=(b'1', '2', 3, 'qwe'))
    assert isinstance(model.event, Event)
    assert model.event == Event(1, 2, 3, 'qwe')
    assert repr(model) == "Model(event=Event(a=1, b=2, c=3, d='qwe'))"
    with pytest.raises(ValidationError) as exc_info:
        Model(pos=('1', 2), event=['qwe', '2', 3, 'qwe'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': ('event', 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'qwe'}]

def test_namedtuple_schema():
    if False:
        i = 10
        return i + 15

    class Position1(NamedTuple):
        x: int
        y: int
    Position2 = namedtuple('Position2', 'x y')

    class Model(BaseModel):
        pos1: Position1
        pos2: Position2
        pos3: Tuple[int, int]
    assert Model.model_json_schema() == {'title': 'Model', 'type': 'object', '$defs': {'Position1': {'maxItems': 2, 'minItems': 2, 'prefixItems': [{'title': 'X', 'type': 'integer'}, {'title': 'Y', 'type': 'integer'}], 'type': 'array'}, 'Position2': {'maxItems': 2, 'minItems': 2, 'prefixItems': [{'title': 'X'}, {'title': 'Y'}], 'type': 'array'}}, 'properties': {'pos1': {'$ref': '#/$defs/Position1'}, 'pos2': {'$ref': '#/$defs/Position2'}, 'pos3': {'maxItems': 2, 'minItems': 2, 'prefixItems': [{'type': 'integer'}, {'type': 'integer'}], 'title': 'Pos3', 'type': 'array'}}, 'required': ['pos1', 'pos2', 'pos3']}

def test_namedtuple_right_length():
    if False:
        print('Hello World!')

    class Point(NamedTuple):
        x: int
        y: int

    class Model(BaseModel):
        p: Point
    assert isinstance(Model(p=(1, 2)), Model)
    with pytest.raises(ValidationError) as exc_info:
        Model(p=(1, 2, 3))
    assert exc_info.value.errors(include_url=False) == [{'type': 'unexpected_positional_argument', 'loc': ('p', 2), 'msg': 'Unexpected positional argument', 'input': 3}]

def test_namedtuple_postponed_annotation():
    if False:
        i = 10
        return i + 15
    '\n    https://github.com/pydantic/pydantic/issues/2760\n    '

    class Tup(NamedTuple):
        v: 'PositiveInt'

    class Model(BaseModel):
        t: Tup
    with pytest.raises(ValidationError):
        Model.model_validate({'t': [-1]})

def test_namedtuple_arbitrary_type():
    if False:
        print('Hello World!')

    class CustomClass:
        pass

    class Tup(NamedTuple):
        c: CustomClass

    class Model(BaseModel):
        x: Tup
        model_config = ConfigDict(arbitrary_types_allowed=True)
    data = {'x': Tup(c=CustomClass())}
    model = Model.model_validate(data)
    assert isinstance(model.x.c, CustomClass)
    with pytest.raises(PydanticSchemaGenerationError):

        class ModelNoArbitraryTypes(BaseModel):
            x: Tup

def test_recursive_namedtuple():
    if False:
        return 10

    class MyNamedTuple(NamedTuple):
        x: int
        y: Optional['MyNamedTuple']
    ta = TypeAdapter(MyNamedTuple)
    assert ta.validate_python({'x': 1, 'y': {'x': 2, 'y': None}}) == (1, (2, None))
    with pytest.raises(ValidationError) as exc_info:
        ta.validate_python({'x': 1, 'y': {'x': 2, 'y': {'x': 'a', 'y': None}}})
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('y', 'y', 'x'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_recursive_generic_namedtuple():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    class MyNamedTuple(TypingExtensionsNamedTuple, Generic[T]):
        x: T
        y: Optional['MyNamedTuple[T]']
    ta = TypeAdapter(MyNamedTuple[int])
    assert ta.validate_python({'x': 1, 'y': {'x': 2, 'y': None}}) == (1, (2, None))
    with pytest.raises(ValidationError) as exc_info:
        ta.validate_python({'x': 1, 'y': {'x': 2, 'y': {'x': 'a', 'y': None}}})
    assert exc_info.value.errors(include_url=False) == [{'input': 'a', 'loc': ('y', 'y', 'x'), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

def test_namedtuple_defaults():
    if False:
        return 10

    class NT(NamedTuple):
        x: int
        y: int = 33
    assert TypeAdapter(NT).validate_python([1]) == (1, 33)
    assert TypeAdapter(NT).validate_python({'x': 22}) == (22, 33)