import textwrap
import strawberry
from strawberry.schema.config import StrawberryConfig
from strawberry.schema_directive import Location

def test_additional_schema_directives_printed_correctly_object():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.OBJECT])
    class CacheControl:
        max_age: int

    @strawberry.federation.type(keys=['id'], shareable=True, extend=True, directives=[CacheControl(max_age=42)])
    class FederatedType:
        id: strawberry.ID

    @strawberry.type
    class Query:
        federatedType: FederatedType
    expected_type = '\n    directive @CacheControl(max_age: Int!) on OBJECT\n\n    extend type FederatedType @CacheControl(max_age: 42) @key(fields: "id") @shareable {\n      id: ID!\n    }\n\n    type Query {\n      federatedType: FederatedType!\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert schema.as_str() == textwrap.dedent(expected_type).strip()

def test_additional_schema_directives_printed_in_order_object():
    if False:
        return 10

    @strawberry.schema_directive(locations=[Location.OBJECT])
    class CacheControl0:
        max_age: int

    @strawberry.schema_directive(locations=[Location.OBJECT])
    class CacheControl1:
        min_age: int

    @strawberry.federation.type(keys=['id'], shareable=True, extend=True, directives=[CacheControl0(max_age=42), CacheControl1(min_age=42)])
    class FederatedType:
        id: strawberry.ID

    @strawberry.type
    class Query:
        federatedType: FederatedType
    expected_type = '\n    directive @CacheControl0(max_age: Int!) on OBJECT\n\n    directive @CacheControl1(min_age: Int!) on OBJECT\n\n    extend type FederatedType @CacheControl0(max_age: 42) @CacheControl1(min_age: 42) @key(fields: "id") @shareable {\n      id: ID!\n    }\n\n    type Query {\n      federatedType: FederatedType!\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert schema.as_str() == textwrap.dedent(expected_type).strip()