from enum import Enum
from typing import List, Optional, TypeVar, Union
import strawberry
from strawberry.types.fields.resolver import StrawberryResolver

def test_enum():
    if False:
        print('Hello World!')

    @strawberry.enum
    class Language(Enum):
        ENGLISH = 'english'
        ITALIAN = 'italian'
        JAPANESE = 'japanese'

    def get_spoken_language() -> Language:
        if False:
            print('Hello World!')
        return Language.ENGLISH
    resolver = StrawberryResolver(get_spoken_language)
    assert resolver.type is Language._enum_definition

def test_forward_references():
    if False:
        print('Hello World!')
    global FutureUmpire

    def get_sportsball_official() -> 'FutureUmpire':
        if False:
            return 10
        return FutureUmpire('ref')

    @strawberry.type
    class FutureUmpire:
        name: str
    resolver = StrawberryResolver(get_sportsball_official)
    assert resolver.type is FutureUmpire
    del FutureUmpire

def test_list():
    if False:
        return 10

    def get_collection_types() -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['list', 'tuple', 'dict', 'set']
    resolver = StrawberryResolver(get_collection_types)
    assert resolver.type == List[str]

def test_literal():
    if False:
        return 10

    def version() -> float:
        if False:
            while True:
                i = 10
        return 1.0
    resolver = StrawberryResolver(version)
    assert resolver.type is float

def test_object():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Polygon:
        edges: int
        faces: int

    def get_2d_object() -> Polygon:
        if False:
            print('Hello World!')
        return Polygon(12, 6)
    resolver = StrawberryResolver(get_2d_object)
    assert resolver.type is Polygon

def test_optional():
    if False:
        i = 10
        return i + 15

    def stock_market_tool() -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        ...
    resolver = StrawberryResolver(stock_market_tool)
    assert resolver.type == Optional[str]

def test_type_var():
    if False:
        for i in range(10):
            print('nop')
    T = TypeVar('T')

    def caffeinated_drink() -> T:
        if False:
            return 10
        ...
    resolver = StrawberryResolver(caffeinated_drink)
    assert resolver.type == T

def test_union():
    if False:
        return 10

    @strawberry.type
    class Venn:
        foo: int

    @strawberry.type
    class Diagram:
        bar: float

    def get_overlap() -> Union[Venn, Diagram]:
        if False:
            i = 10
            return i + 15
        ...
    resolver = StrawberryResolver(get_overlap)
    assert resolver.type == Union[Venn, Diagram]