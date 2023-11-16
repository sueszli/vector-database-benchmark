import dataclasses
import re
import sys
import typing
from types import SimpleNamespace
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.errors import InvalidArgument, Unsatisfiable
from hypothesis.internal.reflection import convert_positional_arguments, get_pretty_function_description
from hypothesis.strategies import from_type
from tests.common.debug import find_any
from tests.common.utils import fails_with, temp_registered

@given(st.data())
def test_typing_Final(data):
    if False:
        i = 10
        return i + 15
    value = data.draw(from_type(typing.Final[int]))
    assert isinstance(value, int)

@pytest.mark.parametrize('value', ['dog', b'goldfish', 42, 63.4, -80.5, False])
def test_typing_Literal(value):
    if False:
        return 10
    assert from_type(typing.Literal[value]).example() == value

@given(st.data())
def test_typing_Literal_nested(data):
    if False:
        while True:
            i = 10
    lit = typing.Literal
    values = [(lit['hamster', 0], ('hamster', 0)), (lit[26, False, 'bunny', 130], (26, False, 'bunny', 130)), (lit[lit[1]], {1}), (lit[lit[1], 2], {1, 2}), (lit[1, lit[2], 3], {1, 2, 3}), (lit[lit[lit[1], lit[2]], lit[lit[3], lit[4]]], {1, 2, 3, 4})]
    (literal_type, flattened_literals) = data.draw(st.sampled_from(values))
    assert data.draw(st.from_type(literal_type)) in flattened_literals

class A(typing.TypedDict):
    a: int

@given(from_type(A))
def test_simple_typeddict(value):
    if False:
        i = 10
        return i + 15
    assert type(value) == dict
    assert set(value) == {'a'}
    assert isinstance(value['a'], int)

class B(A, total=False):
    b: bool

@given(from_type(B))
def test_typeddict_with_optional(value):
    if False:
        i = 10
        return i + 15
    assert type(value) == dict
    assert set(value).issubset({'a', 'b'})
    assert isinstance(value['a'], int)
    if 'b' in value:
        assert isinstance(value['b'], bool)
if sys.version_info[:2] < (3, 9):
    xfail_on_38 = pytest.mark.xfail(raises=Unsatisfiable)
else:

    def xfail_on_38(f):
        if False:
            while True:
                i = 10
        return f

@xfail_on_38
def test_simple_optional_key_is_optional():
    if False:
        for i in range(10):
            print('nop')
    find_any(from_type(B), lambda d: 'b' not in d)

class C(B):
    c: str

@given(from_type(C))
def test_typeddict_with_optional_then_required_again(value):
    if False:
        i = 10
        return i + 15
    assert type(value) == dict
    assert set(value).issubset({'a', 'b', 'c'})
    assert isinstance(value['a'], int)
    if 'b' in value:
        assert isinstance(value['b'], bool)
    assert isinstance(value['c'], str)

class NestedDict(typing.TypedDict):
    inner: A

@given(from_type(NestedDict))
def test_typeddict_with_nested_value(value):
    if False:
        return 10
    assert type(value) == dict
    assert set(value) == {'inner'}
    assert isinstance(value['inner']['a'], int)

@xfail_on_38
def test_layered_optional_key_is_optional():
    if False:
        while True:
            i = 10
    find_any(from_type(C), lambda d: 'b' not in d)

@dataclasses.dataclass()
class Node:
    left: typing.Union['Node', int]
    right: typing.Union['Node', int]

@given(st.builds(Node))
def test_can_resolve_recursive_dataclass(val):
    if False:
        i = 10
        return i + 15
    assert isinstance(val, Node)

def test_can_register_new_type_for_typeddicts():
    if False:
        print('Hello World!')
    sentinel = object()
    with temp_registered(C, st.just(sentinel)):
        assert st.from_type(C).example() is sentinel

@pytest.mark.parametrize('lam,source', [(lambda a, /, b: a, 'lambda a, /, b: a'), (lambda a=None, /, b=None: a, 'lambda a=None, /, b=None: a')])
def test_posonly_lambda_formatting(lam, source):
    if False:
        print('Hello World!')
    assert get_pretty_function_description(lam) == source

def test_does_not_convert_posonly_to_keyword():
    if False:
        print('Hello World!')
    (args, kws) = convert_positional_arguments(lambda x, /: None, (1,), {})
    assert args
    assert not kws

@given(x=st.booleans())
def test_given_works_with_keyword_only_params(*, x):
    if False:
        print('Hello World!')
    pass

def test_given_works_with_keyword_only_params_some_unbound():
    if False:
        while True:
            i = 10

    @given(x=st.booleans())
    def test(*, x, y):
        if False:
            print('Hello World!')
        assert y is None
    test(y=None)

def test_given_works_with_positional_only_params():
    if False:
        i = 10
        return i + 15

    @given(y=st.booleans())
    def test(x, /, y):
        if False:
            i = 10
            return i + 15
        pass
    test(None)

def test_cannot_pass_strategies_by_position_if_there_are_posonly_args():
    if False:
        i = 10
        return i + 15

    @given(st.booleans())
    def test(x, /, y):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument):
        test(None)

@fails_with(InvalidArgument)
@given(st.booleans())
def test_cannot_pass_strategies_for_posonly_args(x, /):
    if False:
        print('Hello World!')
    pass

@given(y=st.booleans())
def has_posonly_args(x, /, y):
    if False:
        for i in range(10):
            print('nop')
    pass

def test_example_argument_validation():
    if False:
        while True:
            i = 10
    example(y=None)(has_posonly_args)(1)
    with pytest.raises(InvalidArgument, match=re.escape('Cannot pass positional arguments to @example() when decorating a test function which has positional-only parameters.')):
        example(None)(has_posonly_args)(1)
    with pytest.raises(InvalidArgument, match=re.escape("Inconsistent args: @given() got strategies for 'y', but @example() got arguments for 'x'")):
        example(x=None)(has_posonly_args)(1)

class FooProtocol(typing.Protocol):

    def frozzle(self, x):
        if False:
            i = 10
            return i + 15
        pass

class BarProtocol(typing.Protocol):

    def bazzle(self, y):
        if False:
            print('Hello World!')
        pass

@given(st.data())
def test_can_resolve_registered_protocol(data):
    if False:
        return 10
    with temp_registered(FooProtocol, st.builds(SimpleNamespace, frozzle=st.functions(like=lambda x: ...))):
        obj = data.draw(st.from_type(FooProtocol))
    assert obj.frozzle(x=1) is None

def test_cannot_resolve_un_registered_protocol():
    if False:
        print('Hello World!')
    msg = 'Instance and class checks can only be used with @runtime_checkable protocols'
    with pytest.raises(TypeError, match=msg):
        st.from_type(BarProtocol).example()