import uuid
from graphql import GraphQLError
import strawberry

def test_uuid():
    if False:
        return 10

    @strawberry.type
    class Query:

        @strawberry.field
        def example_uuid_out(self) -> uuid.UUID:
            if False:
                print('Hello World!')
            return uuid.NAMESPACE_DNS
    schema = strawberry.Schema(Query)
    result = schema.execute_sync('{ exampleUuidOut }')
    assert not result.errors
    assert result.data['exampleUuidOut'] == str(uuid.NAMESPACE_DNS)

def test_uuid_as_input():
    if False:
        return 10

    @strawberry.type
    class Query:

        @strawberry.field
        def example_uuid_in(self, uid: uuid.UUID) -> uuid.UUID:
            if False:
                for i in range(10):
                    print('nop')
            return uid
    schema = strawberry.Schema(Query)
    result = schema.execute_sync(f'{{ exampleUuidIn(uid: "{uuid.NAMESPACE_DNS!s}") }}')
    assert not result.errors
    assert result.data['exampleUuidIn'] == str(uuid.NAMESPACE_DNS)

def test_serialization_of_incorrect_uuid_string():
    if False:
        print('Hello World!')
    '\n    Test GraphQLError is raised for an invalid UUID.\n    The error should exclude "original_error".\n    '

    @strawberry.type
    class Query:
        ok: bool

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def uuid_input(self, uuid_input: uuid.UUID) -> uuid.UUID:
            if False:
                print('Hello World!')
            return uuid_input
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    result = schema.execute_sync('\n            mutation uuidInput($value: UUID!) {\n                uuidInput(uuidInput: $value)\n            }\n        ', variable_values={'value': 'fail'})
    assert result.errors
    assert isinstance(result.errors[0], GraphQLError)
    assert result.errors[0].original_error is None
    assert result.errors[0].message == 'Variable \'$value\' got invalid value \'fail\'; Value cannot represent a UUID: "fail". badly formed hexadecimal UUID string'