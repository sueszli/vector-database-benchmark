import textwrap
import strawberry
from strawberry.printer import print_schema

def test_cyclic_import():
    if False:
        print('Hello World!')
    from .type_a import TypeA
    from .type_b import TypeB

    @strawberry.type
    class Query:
        a: TypeA
        b: TypeB
    expected = '\n    type Query {\n      a: TypeA!\n      b: TypeB!\n    }\n\n    type TypeA {\n      listOfB: [TypeB!]\n      typeB: TypeB!\n    }\n\n    type TypeB {\n      typeA: TypeA!\n    }\n    '
    schema = strawberry.Schema(Query)
    assert print_schema(schema) == textwrap.dedent(expected).strip()