from __future__ import annotations
import textwrap
from typing import Optional
import pydantic
import strawberry

def test_auto_fields():
    if False:
        print('Hello World!')
    global User

    class UserModel(pydantic.BaseModel):
        age: int
        password: Optional[str]
        other: float

    @strawberry.experimental.pydantic.type(UserModel)
    class User:
        age: strawberry.auto
        password: strawberry.auto

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> User:
            if False:
                i = 10
                return i + 15
            return User(age=1, password='ABC')
    schema = strawberry.Schema(query=Query)
    expected_schema = '\n    type Query {\n      user: User!\n    }\n\n    type User {\n      age: Int!\n      password: String\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()
    query = '{ user { age } }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data['user']['age'] == 1