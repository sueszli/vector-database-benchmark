import textwrap
import warnings
from typing import Generic, List, Optional, TypeVar
import pytest
import strawberry

def test_entities_type_when_no_type_has_keys():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.federation.type()
    class Product:
        upc: str
        name: Optional[str]
        price: Optional[int]
        weight: Optional[int]

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                while True:
                    i = 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            __type(name: "_Entity") {\n                kind\n                possibleTypes {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'__type': None}

def test_entities_type():
    if False:
        while True:
            i = 10

    @strawberry.federation.type(keys=['upc'])
    class Product:
        upc: str
        name: Optional[str]
        price: Optional[int]
        weight: Optional[int]

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                return 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            __type(name: "_Entity") {\n                kind\n                possibleTypes {\n                    name\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'__type': {'kind': 'UNION', 'possibleTypes': [{'name': 'Product'}]}}

def test_additional_scalars():
    if False:
        print('Hello World!')

    @strawberry.federation.type(keys=['upc'])
    class Example:
        upc: str

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Example]:
            if False:
                for i in range(10):
                    print('nop')
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            __type(name: "_Any") {\n                kind\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'__type': {'kind': 'SCALAR'}}

def test_service():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.type
    class Product:
        upc: str

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                return 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            _service {\n                sdl\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    sdl = '\n        type Product {\n          upc: String!\n        }\n\n        extend type Query {\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert result.data == {'_service': {'sdl': textwrap.dedent(sdl).strip()}}

def test_using_generics():
    if False:
        return 10
    T = TypeVar('T')

    @strawberry.federation.type
    class Product:
        upc: str

    @strawberry.type
    class ListOfProducts(Generic[T]):
        products: List[T]

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> ListOfProducts[Product]:
            if False:
                i = 10
                return i + 15
            return ListOfProducts(products=[])
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            _service {\n                sdl\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    sdl = '\n        type Product {\n          upc: String!\n        }\n\n        type ProductListOfProducts {\n          products: [Product!]!\n        }\n\n        extend type Query {\n          _service: _Service!\n          topProducts(first: Int!): ProductListOfProducts!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert result.data == {'_service': {'sdl': textwrap.dedent(sdl).strip()}}

def test_input_types():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.input(inaccessible=True)
    class ExampleInput:
        upc: str

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, example: ExampleInput) -> List[str]:
            if False:
                while True:
                    i = 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query {\n            __type(name: "ExampleInput") {\n                kind\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'__type': {'kind': 'INPUT_OBJECT'}}

def test_can_create_schema_without_query():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.type()
    class Product:
        upc: str
        name: Optional[str]
        price: Optional[int]
        weight: Optional[int]
    schema = strawberry.federation.Schema(types=[Product], enable_federation_2=True)
    assert str(schema) == textwrap.dedent('\n                type Product {\n                  upc: String!\n                  name: String\n                  price: Int\n                  weight: Int\n                }\n\n                type Query {\n                  _service: _Service!\n                }\n\n                scalar _Any\n\n                type _Service {\n                  sdl: String!\n                }\n            ').strip()

def test_federation_schema_warning():
    if False:
        return 10

    @strawberry.federation.type(keys=['upc'])
    class ProductFed:
        upc: str
        name: Optional[str]
        price: Optional[int]
        weight: Optional[int]
    with pytest.warns(UserWarning) as record:
        strawberry.Schema(query=ProductFed)
    assert 'Federation directive found in schema. Use `strawberry.federation.Schema` instead of `strawberry.Schema`.' in [str(r.message) for r in record]

def test_does_not_warn_when_using_federation_schema():
    if False:
        print('Hello World!')

    @strawberry.federation.type(keys=['upc'])
    class ProductFed:
        upc: str
        name: Optional[str]
        price: Optional[int]
        weight: Optional[int]

    @strawberry.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[ProductFed]:
            if False:
                for i in range(10):
                    print('nop')
            return []
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('ignore', category=DeprecationWarning, message="'.*' is deprecated and slated for removal in Python 3\\.\\d+")
        strawberry.federation.Schema(query=Query, enable_federation_2=True)
    assert len(w) == 0