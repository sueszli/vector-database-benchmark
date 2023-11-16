import typing
from pydantic import BaseModel
import strawberry
from strawberry.federation.schema_directives import Key

def test_fetch_entities_pydantic():
    if False:
        for i in range(10):
            print('nop')

    class ProductInDb(BaseModel):
        upc: str
        name: str

    @strawberry.experimental.pydantic.type(model=ProductInDb, directives=[Key(fields='upc', resolvable=True)])
    class Product:
        upc: str
        name: str

        @classmethod
        def resolve_reference(cls, upc) -> 'Product':
            if False:
                i = 10
                return i + 15
            return Product(upc=upc, name='')

    @strawberry.federation.type(extend=True)
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> typing.List[Product]:
            if False:
                print('Hello World!')
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    query = '\n        query ($representations: [_Any!]!) {\n            _entities(representations: $representations) {\n                ... on Product {\n                    upc\n                }\n            }\n        }\n    '
    result = schema.execute_sync(query, variable_values={'representations': [{'__typename': 'Product', 'upc': 'B00005N5PF'}]})
    assert not result.errors
    assert result.data == {'_entities': [{'upc': 'B00005N5PF'}]}