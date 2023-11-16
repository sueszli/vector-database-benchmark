import sys
from typing import Any, Type
if sys.version_info < (3, 9):
    from typing_extensions import Annotated
else:
    from typing import Annotated
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

@pytest.fixture(scope='session', name='ModelWithStrictField')
def model_with_strict_field():
    if False:
        i = 10
        return i + 15

    class ModelWithStrictField(BaseModel):
        a: Annotated[int, Field(strict=True)]
    return ModelWithStrictField

@pytest.mark.parametrize('value', ['1', True, 1.0])
def test_parse_strict_mode_on_field_invalid(value: Any, ModelWithStrictField: Type[BaseModel]) -> None:
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValidationError) as exc_info:
        ModelWithStrictField(a=value)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('a',), 'msg': 'Input should be a valid integer', 'input': value}]

def test_parse_strict_mode_on_field_valid(ModelWithStrictField: Type[BaseModel]) -> None:
    if False:
        i = 10
        return i + 15
    value = ModelWithStrictField(a=1)
    assert value.model_dump() == {'a': 1}

@pytest.fixture(scope='session', name='ModelWithStrictConfig')
def model_with_strict_config_false():
    if False:
        for i in range(10):
            print('nop')

    class ModelWithStrictConfig(BaseModel):
        a: int
        b: Annotated[int, Field(strict=False)]
        c: Annotated[int, Field(strict=None)]
        d: Annotated[int, Field()]
        model_config = ConfigDict(strict=True)
    return ModelWithStrictConfig

def test_parse_model_with_strict_config_enabled(ModelWithStrictConfig: Type[BaseModel]) -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValidationError) as exc_info:
        ModelWithStrictConfig(a='1', b=2, c=3, d=4)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('a',), 'msg': 'Input should be a valid integer', 'input': '1'}]
    with pytest.raises(ValidationError) as exc_info:
        ModelWithStrictConfig(a=1, b=2, c='3', d=4)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('c',), 'msg': 'Input should be a valid integer', 'input': '3'}]
    with pytest.raises(ValidationError) as exc_info:
        ModelWithStrictConfig(a=1, b=2, c=3, d='4')
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': ('d',), 'msg': 'Input should be a valid integer', 'input': '4'}]
    values = [ModelWithStrictConfig(a=1, b='2', c=3, d=4), ModelWithStrictConfig(a=1, b=2, c=3, d=4)]
    assert all((v.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4} for v in values))

def test_parse_model_with_strict_config_disabled(ModelWithStrictConfig: Type[BaseModel]) -> None:
    if False:
        while True:
            i = 10

    class Model(ModelWithStrictConfig):
        model_config = ConfigDict(strict=False)
    values = [Model(a='1', b=2, c=3, d=4), Model(a=1, b=2, c='3', d=4), Model(a=1, b=2, c=3, d='4'), Model(a=1, b='2', c=3, d=4), Model(a=1, b=2, c=3, d=4)]
    assert all((v.model_dump() == {'a': 1, 'b': 2, 'c': 3, 'd': 4} for v in values))