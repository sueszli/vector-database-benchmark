from __future__ import annotations
import textwrap
from uuid import UUID
import strawberry
from strawberry.field_extensions import InputMutationExtension

@strawberry.type
class Query:

    @strawberry.field
    async def hello(self) -> str:
        return 'hi'

@strawberry.type
class Mutation:

    @strawberry.mutation(extensions=[InputMutationExtension()])
    async def buggy(self, some_id: UUID) -> None:
        del some_id

def test_schema():
    if False:
        i = 10
        return i + 15
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    expected_schema = '\n    input BuggyInput {\n      someId: UUID!\n    }\n\n    type Mutation {\n      buggy(\n        """Input data for `buggy` mutation"""\n        input: BuggyInput!\n      ): Void\n    }\n\n    type Query {\n      hello: String!\n    }\n\n    scalar UUID\n\n    """Represents NULL values"""\n    scalar Void\n    '
    assert textwrap.dedent(expected_schema).strip() == str(schema).strip()