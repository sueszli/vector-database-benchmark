import dataclasses
import sys
import typing
import pytest
from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument
from tests.common.debug import assert_all_examples, find_any
from tests.common.utils import temp_registered

@pytest.mark.parametrize('annotated_type,expected_strategy_repr', [(typing.Annotated[int, 'foo'], 'integers()'), (typing.Annotated[typing.List[float], 'foo'], 'lists(floats())'), (typing.Annotated[typing.Annotated[str, 'foo'], 'bar'], 'text()'), (typing.Annotated[typing.Annotated[typing.List[typing.Dict[str, bool]], 'foo'], 'bar'], 'lists(dictionaries(keys=text(), values=booleans()))')])
def test_typing_Annotated(annotated_type, expected_strategy_repr):
    if False:
        i = 10
        return i + 15
    assert repr(st.from_type(annotated_type)) == expected_strategy_repr
PositiveInt = typing.Annotated[int, st.integers(min_value=1)]
MoreThenTenInt = typing.Annotated[PositiveInt, st.integers(min_value=10 + 1)]
WithTwoStrategies = typing.Annotated[int, st.integers(), st.none()]
ExtraAnnotationNoStrategy = typing.Annotated[PositiveInt, 'metadata']

def arg_positive(x: PositiveInt):
    if False:
        print('Hello World!')
    assert x > 0

def arg_more_than_ten(x: MoreThenTenInt):
    if False:
        print('Hello World!')
    assert x > 10

@given(st.data())
def test_annotated_positive_int(data):
    if False:
        return 10
    data.draw(st.builds(arg_positive))

@given(st.data())
def test_annotated_more_than_ten(data):
    if False:
        print('Hello World!')
    data.draw(st.builds(arg_more_than_ten))

@given(st.data())
def test_annotated_with_two_strategies(data):
    if False:
        return 10
    assert data.draw(st.from_type(WithTwoStrategies)) is None

@given(st.data())
def test_annotated_extra_metadata(data):
    if False:
        for i in range(10):
            print('nop')
    assert data.draw(st.from_type(ExtraAnnotationNoStrategy)) > 0

@dataclasses.dataclass
class User:
    id: int
    following: list['User']

@pytest.mark.skipif(sys.version_info[:2] >= (3, 11), reason='works in new Pythons')
def test_string_forward_ref_message():
    if False:
        return 10
    s = st.builds(User)
    with pytest.raises(InvalidArgument, match='`from __future__ import annotations`'):
        s.example()

def test_issue_3080():
    if False:
        print('Hello World!')
    s = st.from_type(typing.Union[list[int], int])
    find_any(s, lambda x: isinstance(x, int))
    find_any(s, lambda x: isinstance(x, list))

@dataclasses.dataclass
class TypingTuple:
    a: dict[typing.Tuple[int, int], str]

@dataclasses.dataclass
class BuiltinTuple:
    a: dict[tuple[int, int], str]
TestDataClass = typing.Union[TypingTuple, BuiltinTuple]

@pytest.mark.parametrize('data_class', [TypingTuple, BuiltinTuple])
@given(data=st.data())
def test_from_type_with_tuple_works(data, data_class: TestDataClass):
    if False:
        i = 10
        return i + 15
    value: TestDataClass = data.draw(st.from_type(data_class))
    assert len(value.a) >= 0

def _shorter_lists(list_type):
    if False:
        for i in range(10):
            print('nop')
    return st.lists(st.from_type(*typing.get_args(list_type)), max_size=2)

def test_can_register_builtin_list():
    if False:
        print('Hello World!')
    with temp_registered(list, _shorter_lists):
        assert_all_examples(st.from_type(list[int]), lambda ls: len(ls) <= 2 and {type(x) for x in ls}.issubset({int}))
T = typing.TypeVar('T')

@typing.runtime_checkable
class Fooable(typing.Protocol[T]):

    def foo(self):
        if False:
            i = 10
            return i + 15
        ...

class FooableConcrete(tuple):

    def foo(self):
        if False:
            print('Hello World!')
        pass

def test_only_tuple_subclasses_in_typing_type():
    if False:
        print('Hello World!')
    with temp_registered(FooableConcrete, st.builds(FooableConcrete)):
        s = st.from_type(Fooable[int])
        assert_all_examples(s, lambda x: type(x) is FooableConcrete)

def test_lookup_registered_tuple():
    if False:
        while True:
            i = 10
    sentinel = object()
    typ = tuple[int]
    with temp_registered(tuple, st.just(sentinel)):
        assert st.from_type(typ).example() is sentinel
    assert st.from_type(typ).example() is not sentinel