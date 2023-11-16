import asyncio
import inspect
import re
import sys
from datetime import datetime, timezone
from functools import partial
from typing import Any, List, Tuple
import pytest
from pydantic_core import ArgsKwargs
from typing_extensions import Annotated, TypedDict
from pydantic import Field, PydanticInvalidForJsonSchema, TypeAdapter, ValidationError, validate_call
from pydantic.main import BaseModel
skip_pre_39 = pytest.mark.skipif(sys.version_info < (3, 9), reason='testing >= 3.9 behaviour only')

def test_args():
    if False:
        return 10

    @validate_call
    def foo(a: int, b: int):
        if False:
            while True:
                i = 10
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(*[1, 2]) == '1, 2'
    assert foo(*(1, 2)) == '1, 2'
    assert foo(*[1], 2) == '1, 2'
    assert foo(a=1, b=2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo(b=2, a=1) == '1, 2'
    with pytest.raises(ValidationError) as exc_info:
        foo()
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (1,), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]
    with pytest.raises(ValidationError, match='2\\s+Unexpected positional argument'):
        foo(1, 2, 3)
    with pytest.raises(ValidationError, match='apple\\s+Unexpected keyword argument'):
        foo(1, 2, apple=3)
    with pytest.raises(ValidationError, match='a\\s+Got multiple values for argument'):
        foo(1, 2, a=3)
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, a=3, b=4)
    assert exc_info.value.errors(include_url=False) == [{'type': 'multiple_argument_values', 'loc': ('a',), 'msg': 'Got multiple values for argument', 'input': 3}, {'type': 'multiple_argument_values', 'loc': ('b',), 'msg': 'Got multiple values for argument', 'input': 4}]

def test_optional():
    if False:
        for i in range(10):
            print('nop')

    @validate_call
    def foo_bar(a: int=None):
        if False:
            i = 10
            return i + 15
        return f'a={a}'
    assert foo_bar() == 'a=None'
    assert foo_bar(1) == 'a=1'
    with pytest.raises(ValidationError) as exc_info:
        foo_bar(None)
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_type', 'loc': (0,), 'msg': 'Input should be a valid integer', 'input': None}]

def test_wrap():
    if False:
        return 10

    @validate_call
    def foo_bar(a: int, b: int):
        if False:
            for i in range(10):
                print('nop')
        'This is the foo_bar method.'
        return f'{a}, {b}'
    assert foo_bar.__doc__ == 'This is the foo_bar method.'
    assert foo_bar.__name__ == 'foo_bar'
    assert foo_bar.__module__ == 'tests.test_validate_call'
    assert foo_bar.__qualname__ == 'test_wrap.<locals>.foo_bar'
    assert isinstance(foo_bar.__pydantic_core_schema__, dict)
    assert callable(foo_bar.raw_function)
    assert repr(foo_bar) == f'ValidateCallWrapper({repr(foo_bar.raw_function)})'
    assert repr(inspect.signature(foo_bar)) == '<Signature (a: int, b: int)>'

def test_kwargs():
    if False:
        print('Hello World!')

    @validate_call
    def foo(*, a: int, b: int):
        if False:
            print('Hello World!')
        return a + b
    assert foo(a=1, b=3) == 4
    with pytest.raises(ValidationError) as exc_info:
        foo(a=1, b='x')
    assert exc_info.value.errors(include_url=False) == [{'input': 'x', 'loc': ('b',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 'x')
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_keyword_only_argument', 'loc': ('a',), 'msg': 'Missing required keyword only argument', 'input': ArgsKwargs((1, 'x'))}, {'type': 'missing_keyword_only_argument', 'loc': ('b',), 'msg': 'Missing required keyword only argument', 'input': ArgsKwargs((1, 'x'))}, {'type': 'unexpected_positional_argument', 'loc': (0,), 'msg': 'Unexpected positional argument', 'input': 1}, {'type': 'unexpected_positional_argument', 'loc': (1,), 'msg': 'Unexpected positional argument', 'input': 'x'}]

def test_untyped():
    if False:
        for i in range(10):
            print('nop')

    @validate_call
    def foo(a, b, c='x', *, d='y'):
        if False:
            return 10
        return ', '.join((str(arg) for arg in [a, b, c, d]))
    assert foo(1, 2) == '1, 2, x, y'
    assert foo(1, {'x': 2}, c='3', d='4') == "1, {'x': 2}, 3, 4"

@pytest.mark.parametrize('validated', (True, False))
def test_var_args_kwargs(validated):
    if False:
        while True:
            i = 10

    def foo(a, b, *args, d=3, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return f'a={a!r}, b={b!r}, args={args!r}, d={d!r}, kwargs={kwargs!r}'
    if validated:
        foo = validate_call(foo)
    assert foo(1, 2) == 'a=1, b=2, args=(), d=3, kwargs={}'
    assert foo(1, 2, 3, d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(*[1, 2, 3], d=4) == 'a=1, b=2, args=(3,), d=4, kwargs={}'
    assert foo(1, 2, args=(10, 11)) == "a=1, b=2, args=(), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, args=(10, 11)) == "a=1, b=2, args=(3,), d=3, kwargs={'args': (10, 11)}"
    assert foo(1, 2, 3, e=10) == "a=1, b=2, args=(3,), d=3, kwargs={'e': 10}"
    assert foo(1, 2, kwargs=4) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4}"
    assert foo(1, 2, kwargs=4, e=5) == "a=1, b=2, args=(), d=3, kwargs={'kwargs': 4, 'e': 5}"

def test_field_can_provide_factory() -> None:
    if False:
        while True:
            i = 10

    @validate_call
    def foo(a: int, b: int=Field(default_factory=lambda : 99), *args: int) -> int:
        if False:
            while True:
                i = 10
        'mypy is happy with this'
        return a + b + sum(args)
    assert foo(3) == 102
    assert foo(1, 2, 3) == 6

def test_annotated_field_can_provide_factory() -> None:
    if False:
        print('Hello World!')

    @validate_call
    def foo2(a: int, b: Annotated[int, Field(default_factory=lambda : 99)], *args: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        'mypy reports Incompatible default for argument "b" if we don\'t supply ANY as default'
        return a + b + sum(args)
    assert foo2(1) == 100

def test_positional_only(create_module):
    if False:
        for i in range(10):
            print('nop')
    module = create_module("\nfrom pydantic import validate_call\n\n@validate_call\ndef foo(a, b, /, c=None):\n    return f'{a}, {b}, {c}'\n")
    assert module.foo(1, 2) == '1, 2, None'
    assert module.foo(1, 2, 44) == '1, 2, 44'
    assert module.foo(1, 2, c=44) == '1, 2, 44'
    with pytest.raises(ValidationError) as exc_info:
        module.foo(1, b=2)
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_positional_only_argument', 'loc': (1,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((1,), {'b': 2})}, {'type': 'unexpected_keyword_argument', 'loc': ('b',), 'msg': 'Unexpected keyword argument', 'input': 2}]
    with pytest.raises(ValidationError) as exc_info:
        module.foo(a=1, b=2)
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_positional_only_argument', 'loc': (0,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((), {'a': 1, 'b': 2})}, {'type': 'missing_positional_only_argument', 'loc': (1,), 'msg': 'Missing required positional only argument', 'input': ArgsKwargs((), {'a': 1, 'b': 2})}, {'type': 'unexpected_keyword_argument', 'loc': ('a',), 'msg': 'Unexpected keyword argument', 'input': 1}, {'type': 'unexpected_keyword_argument', 'loc': ('b',), 'msg': 'Unexpected keyword argument', 'input': 2}]

def test_args_name():
    if False:
        return 10

    @validate_call
    def foo(args: int, kwargs: int):
        if False:
            print('Hello World!')
        return f'args={args!r}, kwargs={kwargs!r}'
    assert foo(1, 2) == 'args=1, kwargs=2'
    with pytest.raises(ValidationError, match='apple\\s+Unexpected keyword argument'):
        foo(1, 2, apple=4)
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, apple=4, banana=5)
    assert exc_info.value.errors(include_url=False) == [{'type': 'unexpected_keyword_argument', 'loc': ('apple',), 'msg': 'Unexpected keyword argument', 'input': 4}, {'type': 'unexpected_keyword_argument', 'loc': ('banana',), 'msg': 'Unexpected keyword argument', 'input': 5}]
    with pytest.raises(ValidationError) as exc_info:
        foo(1, 2, 3)
    assert exc_info.value.errors(include_url=False) == [{'type': 'unexpected_positional_argument', 'loc': (2,), 'msg': 'Unexpected positional argument', 'input': 3}]

def test_v_args():
    if False:
        for i in range(10):
            print('nop')

    @validate_call
    def foo1(v__args: int):
        if False:
            while True:
                i = 10
        return v__args
    assert foo1(123) == 123

    @validate_call
    def foo2(v__kwargs: int):
        if False:
            while True:
                i = 10
        return v__kwargs
    assert foo2(123) == 123

    @validate_call
    def foo3(v__positional_only: int):
        if False:
            return 10
        return v__positional_only
    assert foo3(123) == 123

    @validate_call
    def foo4(v__duplicate_kwargs: int):
        if False:
            while True:
                i = 10
        return v__duplicate_kwargs
    assert foo4(123) == 123

def test_async():
    if False:
        i = 10
        return i + 15

    @validate_call
    async def foo(a, b):
        return f'a={a} b={b}'

    async def run():
        v = await foo(1, 2)
        assert v == 'a=1 b=2'
    asyncio.run(run())
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(foo('x'))
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(('x',))}]

def test_string_annotation():
    if False:
        while True:
            i = 10

    @validate_call
    def foo(a: 'List[int]', b: 'float'):
        if False:
            print('Hello World!')
        return f'a={a!r} b={b!r}'
    assert foo([1, 2, 3], 22) == 'a=[1, 2, 3] b=22.0'
    with pytest.raises(ValidationError) as exc_info:
        foo(['x'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (0, 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((['x'],))}]

def test_local_annotation():
    if False:
        return 10
    ListInt = List[int]

    @validate_call
    def foo(a: ListInt):
        if False:
            i = 10
            return i + 15
        return f'a={a!r}'
    assert foo([1, 2, 3]) == 'a=[1, 2, 3]'
    with pytest.raises(ValidationError) as exc_info:
        foo(['x'])
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (0, 0), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]

def test_item_method():
    if False:
        return 10

    class X:

        def __init__(self, v):
            if False:
                i = 10
                return i + 15
            self.v = v

        @validate_call
        def foo(self, a: int, b: int):
            if False:
                print('Hello World!')
            assert self.v == a
            return f'{a}, {b}'
    x = X(4)
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}]

@skip_pre_39
def test_class_method():
    if False:
        i = 10
        return i + 15

    class X:

        @classmethod
        @validate_call
        def foo(cls, a: int, b: int):
            if False:
                for i in range(10):
                    print('nop')
            assert cls == X
            return f'{a}, {b}'
    x = X()
    assert x.foo(4, 2) == '4, 2'
    assert x.foo(*[4, 2]) == '4, 2'
    with pytest.raises(ValidationError) as exc_info:
        x.foo()
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('a',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}, {'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs(())}]

def test_json_schema():
    if False:
        print('Hello World!')

    @validate_call
    def foo(a: int, b: int=None):
        if False:
            for i in range(10):
                print('nop')
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(1, b=2) == '1, 2'
    assert foo(1) == '1, None'
    assert TypeAdapter(foo).json_schema() == {'type': 'object', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'default': None, 'title': 'B', 'type': 'integer'}}, 'required': ['a'], 'additionalProperties': False}

    @validate_call
    def foo(a: int, /, b: int):
        if False:
            for i in range(10):
                print('nop')
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert TypeAdapter(foo).json_schema() == {'maxItems': 2, 'minItems': 2, 'prefixItems': [{'title': 'A', 'type': 'integer'}, {'title': 'B', 'type': 'integer'}], 'type': 'array'}

    @validate_call
    def foo(a: int, /, *, b: int, c: int):
        if False:
            for i in range(10):
                print('nop')
        return f'{a}, {b}, {c}'
    assert foo(1, b=2, c=3) == '1, 2, 3'
    with pytest.raises(PydanticInvalidForJsonSchema, match='Unable to generate JSON schema for arguments validator with positional-only and keyword-only arguments'):
        TypeAdapter(foo).json_schema()

    @validate_call
    def foo(*numbers: int) -> int:
        if False:
            print('Hello World!')
        return sum(numbers)
    assert foo(1, 2, 3) == 6
    assert TypeAdapter(foo).json_schema() == {'items': {'type': 'integer'}, 'prefixItems': [], 'type': 'array'}

    @validate_call
    def foo(**scores: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ', '.join((f'{k}={v}' for (k, v) in sorted(scores.items())))
    assert foo(a=1, b=2) == 'a=1, b=2'
    assert TypeAdapter(foo).json_schema() == {'additionalProperties': {'type': 'integer'}, 'properties': {}, 'type': 'object'}

    @validate_call
    def foo(a: Annotated[int, Field(..., alias='A')]):
        if False:
            while True:
                i = 10
        return a
    assert foo(1) == 1
    assert TypeAdapter(foo).json_schema() == {'additionalProperties': False, 'properties': {'A': {'title': 'A', 'type': 'integer'}}, 'required': ['A'], 'type': 'object'}

def test_alias_generator():
    if False:
        print('Hello World!')

    @validate_call(config=dict(alias_generator=lambda x: x * 2))
    def foo(a: int, b: int):
        if False:
            while True:
                i = 10
        return f'{a}, {b}'
    assert foo(1, 2) == '1, 2'
    assert foo(aa=1, bb=2) == '1, 2'

def test_config_arbitrary_types_allowed():
    if False:
        return 10

    class EggBox:

        def __str__(self) -> str:
            if False:
                i = 10
                return i + 15
            return 'EggBox()'

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def foo(a: int, b: EggBox):
        if False:
            i = 10
            return i + 15
        return f'{a}, {b}'
    assert foo(1, EggBox()) == '1, EggBox()'
    with pytest.raises(ValidationError) as exc_info:
        assert foo(1, 2) == '1, 2'
    assert exc_info.value.errors(include_url=False) == [{'type': 'is_instance_of', 'loc': (1,), 'msg': 'Input should be an instance of test_config_arbitrary_types_allowed.<locals>.EggBox', 'input': 2, 'ctx': {'class': 'test_config_arbitrary_types_allowed.<locals>.EggBox'}}]

def test_annotated_use_of_alias():
    if False:
        while True:
            i = 10

    @validate_call
    def foo(a: Annotated[int, Field(alias='b')], c: Annotated[int, Field()], d: Annotated[int, Field(alias='')]):
        if False:
            for i in range(10):
                print('nop')
        return a + c + d
    assert foo(**{'b': 10, 'c': 12, '': 1}) == 23
    with pytest.raises(ValidationError) as exc_info:
        assert foo(a=10, c=12, d=1) == 10
    assert exc_info.value.errors(include_url=False) == [{'type': 'missing_argument', 'loc': ('b',), 'msg': 'Missing required argument', 'input': ArgsKwargs((), {'a': 10, 'c': 12, 'd': 1})}, {'type': 'missing_argument', 'loc': ('',), 'msg': 'Missing required argument', 'input': ArgsKwargs((), {'a': 10, 'c': 12, 'd': 1})}, {'type': 'unexpected_keyword_argument', 'loc': ('a',), 'msg': 'Unexpected keyword argument', 'input': 10}, {'type': 'unexpected_keyword_argument', 'loc': ('d',), 'msg': 'Unexpected keyword argument', 'input': 1}]

def test_use_of_alias():
    if False:
        for i in range(10):
            print('nop')

    @validate_call
    def foo(c: int=Field(default_factory=lambda : 20), a: int=Field(default_factory=lambda : 10, alias='b')):
        if False:
            i = 10
            return i + 15
        return a + c
    assert foo(b=10) == 30

def test_populate_by_name():
    if False:
        print('Hello World!')

    @validate_call(config=dict(populate_by_name=True))
    def foo(a: Annotated[int, Field(alias='b')], c: Annotated[int, Field(alias='d')]):
        if False:
            return 10
        return a + c
    assert foo(b=10, d=1) == 11
    assert foo(a=10, d=1) == 11
    assert foo(b=10, c=1) == 11
    assert foo(a=10, c=1) == 11

def test_validate_return():
    if False:
        for i in range(10):
            print('nop')

    @validate_call(config=dict(validate_return=True))
    def foo(a: int, b: int) -> int:
        if False:
            return 10
        return a + b
    assert foo(1, 2) == 3

def test_validate_all():
    if False:
        for i in range(10):
            print('nop')

    @validate_call(config=dict(validate_default=True))
    def foo(dt: datetime=Field(default_factory=lambda : 946684800)):
        if False:
            while True:
                i = 10
        return dt
    assert foo() == datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert foo(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)

def test_validate_all_positional(create_module):
    if False:
        print('Hello World!')
    module = create_module('\nfrom datetime import datetime\n\nfrom pydantic import Field, validate_call\n\n@validate_call(config=dict(validate_default=True))\ndef foo(dt: datetime = Field(default_factory=lambda: 946684800), /):\n    return dt\n')
    assert module.foo() == datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert module.foo(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)

def test_partial():
    if False:
        for i in range(10):
            print('nop')

    def my_wrapped_function(a: int, b: int, c: int):
        if False:
            for i in range(10):
                print('nop')
        return a + b + c
    my_partial_function = partial(my_wrapped_function, c=3)
    f = validate_call(my_partial_function)
    assert f(1, 2) == 6

def test_validator_init():
    if False:
        for i in range(10):
            print('nop')

    class Foo:

        @validate_call
        def __init__(self, a: int, b: int):
            if False:
                return 10
            self.v = a + b
    assert Foo(1, 2).v == 3
    assert Foo(1, '2').v == 3
    with pytest.raises(ValidationError, match="type=int_parsing, input_value='x', input_type=str"):
        Foo(1, 'x')

def test_positional_and_keyword_with_same_name(create_module):
    if False:
        i = 10
        return i + 15
    module = create_module('\nfrom pydantic import validate_call\n\n@validate_call\ndef f(a: int, /, **kwargs):\n    return a, kwargs\n')
    assert module.f(1, a=2) == (1, {'a': 2})

def test_model_as_arg() -> None:
    if False:
        print('Hello World!')

    class Model1(TypedDict):
        x: int

    class Model2(BaseModel):
        y: int

    @validate_call(validate_return=True)
    def f1(m1: Model1, m2: Model2) -> Tuple[Model1, Model2]:
        if False:
            while True:
                i = 10
        return (m1, m2.model_dump())
    res = f1({'x': '1'}, {'y': '2'})
    assert res == ({'x': 1}, Model2(y=2))

def test_do_not_call_repr_on_validate_call() -> None:
    if False:
        while True:
            i = 10

    class Class:

        @validate_call
        def __init__(self, number: int) -> None:
            if False:
                i = 10
                return i + 15
            ...

        def __repr__(self) -> str:
            if False:
                return 10
            assert False
    Class(50)

def test_methods_are_not_rebound():
    if False:
        return 10

    class Thing:

        def __init__(self, x: int):
            if False:
                for i in range(10):
                    print('nop')
            self.x = x

        def a(self, x: int):
            if False:
                print('Hello World!')
            return x + self.x
        c = validate_call(a)
    thing = Thing(1)
    assert thing.a == thing.a
    assert thing.c == thing.c
    assert Thing.c == Thing.c
    assert Thing.c(thing, '2') == 3
    assert Thing(2).c('3') == 5

def test_basemodel_method():
    if False:
        for i in range(10):
            print('nop')

    class Foo(BaseModel):

        @classmethod
        @validate_call
        def test(cls, x: int):
            if False:
                return 10
            return (cls, x)
    assert Foo.test('1') == (Foo, 1)

    class Bar(BaseModel):

        @validate_call
        def test(self, x: int):
            if False:
                i = 10
                return i + 15
            return (self, x)
    bar = Bar()
    assert bar.test('1') == (bar, 1)

@pytest.mark.parametrize('decorator', [staticmethod, classmethod])
def test_classmethod_order_error(decorator):
    if False:
        return 10
    name = decorator.__name__
    with pytest.raises(TypeError, match=re.escape(f'The `@{name}` decorator should be applied after `@validate_call` (put `@{name}` on top)')):

        class A:

            @validate_call
            @decorator
            def method(self, x: int):
                if False:
                    for i in range(10):
                        print('nop')
                pass

def test_async_func() -> None:
    if False:
        print('Hello World!')

    @validate_call(validate_return=True)
    async def foo(a: Any) -> int:
        return a
    res = asyncio.run(foo(1))
    assert res == 1
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(foo('x'))
    assert exc_info.value.errors(include_url=False) == [{'type': 'int_parsing', 'loc': (), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'input': 'x'}]

def test_validate_call_with_slots() -> None:
    if False:
        return 10

    class ClassWithSlots:
        __slots__ = {}

        @validate_call(validate_return=True)
        def some_instance_method(self, x: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return x

        @classmethod
        @validate_call(validate_return=True)
        def some_class_method(cls, x: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return x

        @staticmethod
        @validate_call(validate_return=True)
        def some_static_method(x: str) -> str:
            if False:
                print('Hello World!')
            return x
    c = ClassWithSlots()
    assert c.some_instance_method(x='potato') == 'potato'
    assert c.some_class_method(x='pepper') == 'pepper'
    assert c.some_static_method(x='onion') == 'onion'
    assert c.some_instance_method == c.some_instance_method
    assert c.some_class_method == c.some_class_method
    assert c.some_static_method == c.some_static_method