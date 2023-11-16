import textwrap
from typing_extensions import Annotated
import strawberry
from strawberry.field_extensions import InputMutationExtension
from strawberry.schema_directive import Location, schema_directive

@schema_directive(locations=[Location.FIELD_DEFINITION], name='some_directive')
class SomeDirective:
    some: str
    directive: str

@strawberry.type
class Fruit:
    name: str
    color: str

@strawberry.type
class Query:

    @strawberry.field
    def hello(self) -> str:
        if False:
            return 10
        return 'hi'

@strawberry.type
class Mutation:

    @strawberry.mutation(extensions=[InputMutationExtension()])
    def create_fruit(self, name: str, color: Annotated[str, strawberry.argument(description='The color of the fruit', directives=[SomeDirective(some='foo', directive='bar')])]) -> Fruit:
        if False:
            print('Hello World!')
        return Fruit(name=name, color=color)

    @strawberry.mutation(extensions=[InputMutationExtension()])
    async def create_fruit_async(self, name: str, color: Annotated[str, object()]) -> Fruit:
        return Fruit(name=name, color=color)
schema = strawberry.Schema(query=Query, mutation=Mutation)

def test_schema():
    if False:
        i = 10
        return i + 15
    expected = '\n    directive @some_directive(some: String!, directive: String!) on FIELD_DEFINITION\n\n    input CreateFruitAsyncInput {\n      name: String!\n      color: String!\n    }\n\n    input CreateFruitInput {\n      name: String!\n\n      """The color of the fruit"""\n      color: String! @some_directive(some: "foo", directive: "bar")\n    }\n\n    type Fruit {\n      name: String!\n      color: String!\n    }\n\n    type Mutation {\n      createFruit(\n        """Input data for `createFruit` mutation"""\n        input: CreateFruitInput!\n      ): Fruit!\n      createFruitAsync(\n        """Input data for `createFruitAsync` mutation"""\n        input: CreateFruitAsyncInput!\n      ): Fruit!\n    }\n\n    type Query {\n      hello: String!\n    }\n    '
    assert str(schema).strip() == textwrap.dedent(expected).strip()

def test_input_mutation():
    if False:
        return 10
    result = schema.execute_sync('\n        mutation TestQuery ($input: CreateFruitInput!) {\n            createFruit (input: $input) {\n                ... on Fruit {\n                    name\n                    color\n                }\n            }\n        }\n        ', variable_values={'input': {'name': 'Dragonfruit', 'color': 'red'}})
    assert result.errors is None
    assert result.data == {'createFruit': {'name': 'Dragonfruit', 'color': 'red'}}

async def test_input_mutation_async():
    result = await schema.execute('\n        mutation TestQuery ($input: CreateFruitAsyncInput!) {\n            createFruitAsync (input: $input) {\n                ... on Fruit {\n                    name\n                    color\n                }\n            }\n        }\n        ', variable_values={'input': {'name': 'Dragonfruit', 'color': 'red'}})
    assert result.errors is None
    assert result.data == {'createFruitAsync': {'name': 'Dragonfruit', 'color': 'red'}}