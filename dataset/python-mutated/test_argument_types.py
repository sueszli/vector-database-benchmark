import warnings
from enum import Enum
from typing import Any, List, Optional, TypeVar
import pytest
import strawberry
from strawberry.types.info import Info

def test_enum():
    if False:
        print('Hello World!')

    @strawberry.enum
    class Locale(Enum):
        UNITED_STATES = 'en_US'
        UK = 'en_UK'
        AUSTRALIA = 'en_AU'

    @strawberry.mutation
    def set_locale(locale: Locale) -> bool:
        if False:
            i = 10
            return i + 15
        _ = locale
        return True
    argument = set_locale.arguments[0]
    assert argument.type is Locale._enum_definition

def test_forward_reference():
    if False:
        return 10
    global SearchInput

    @strawberry.field
    def search(search_input: 'SearchInput') -> bool:
        if False:
            print('Hello World!')
        _ = search_input
        return True

    @strawberry.input
    class SearchInput:
        query: str
    argument = search.arguments[0]
    assert argument.type is SearchInput
    del SearchInput

def test_list():
    if False:
        while True:
            i = 10

    @strawberry.field
    def get_longest_word(words: List[str]) -> str:
        if False:
            while True:
                i = 10
        _ = words
        return 'I cheated'
    argument = get_longest_word.arguments[0]
    assert argument.type == List[str]

def test_literal():
    if False:
        print('Hello World!')

    @strawberry.field
    def get_name(id_: int) -> str:
        if False:
            while True:
                i = 10
        _ = id_
        return 'Lord Buckethead'
    argument = get_name.arguments[0]
    assert argument.type == int

def test_object():
    if False:
        print('Hello World!')

    @strawberry.type
    class PersonInput:
        proper_noun: bool

    @strawberry.field
    def get_id(person_input: PersonInput) -> int:
        if False:
            while True:
                i = 10
        _ = person_input
        return 0
    argument = get_id.arguments[0]
    assert argument.type is PersonInput

def test_optional():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.field
    def set_age(age: Optional[int]) -> bool:
        if False:
            return 10
        _ = age
        return True
    argument = set_age.arguments[0]
    assert argument.type == Optional[int]

def test_type_var():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.field
    def set_value(value: T) -> bool:
        if False:
            print('Hello World!')
        _ = value
        return True
    argument = set_value.arguments[0]
    assert argument.type == T
ContextType = TypeVar('ContextType')
RootValueType = TypeVar('RootValueType')

class CustomInfo(Info[ContextType, RootValueType]):
    """Subclassed Info type used to test dependency injection."""

@pytest.mark.parametrize('annotation', [CustomInfo, CustomInfo[Any, Any], Info, Info[Any, Any]])
def test_custom_info(annotation):
    if False:
        while True:
            i = 10
    'Test to ensure that subclassed Info does not raise warning.'
    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        def get_info(info) -> bool:
            if False:
                while True:
                    i = 10
            _ = info
            return True
        get_info.__annotations__['info'] = annotation
        get_info_field = strawberry.field(get_info)
        assert not get_info_field.arguments
        info_parameter = get_info_field.base_resolver.info_parameter
        assert info_parameter is not None
        assert info_parameter.name == 'info'

def test_custom_info_negative():
    if False:
        while True:
            i = 10
    'Test to ensure deprecation warning is emitted.'
    with pytest.warns(DeprecationWarning, match="Argument name-based matching of 'info'"):

        @strawberry.field
        def get_info(info) -> bool:
            if False:
                return 10
            _ = info
            return True
        assert not get_info.arguments
        info_parameter = get_info.base_resolver.info_parameter
        assert info_parameter is not None
        assert info_parameter.name == 'info'