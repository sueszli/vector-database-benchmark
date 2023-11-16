import pytest
import strawberry
pytestmark = pytest.mark.pydantic

def test_use_alias_as_gql_name():
    if False:
        print('Hello World!')
    from pydantic import BaseModel, Field

    class UserModel(BaseModel):
        age_: int = Field(..., alias='age_alias')

    @strawberry.experimental.pydantic.type(UserModel, all_fields=True, use_pydantic_alias=True)
    class User:
        ...

    @strawberry.type
    class Query:
        user: User = strawberry.field(default_factory=lambda : User(age_=5))
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename,\n\n            ... on User {\n                age_alias\n            }\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['user'] == {'__typename': 'User', 'age_alias': 5}

def test_do_not_use_alias_as_gql_name():
    if False:
        i = 10
        return i + 15
    from pydantic import BaseModel, Field

    class UserModel(BaseModel):
        age_: int = Field(..., alias='age_alias')

    @strawberry.experimental.pydantic.type(UserModel, all_fields=True, use_pydantic_alias=False)
    class User:
        ...

    @strawberry.type
    class Query:
        user: User = strawberry.field(default_factory=lambda : User(age_=5))
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename,\n\n            ... on User {\n                age_\n            }\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['user'] == {'__typename': 'User', 'age_': 5}