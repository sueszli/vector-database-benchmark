from decimal import Decimal
from graphql import GraphQLError
import strawberry

def test_decimal():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:

        @strawberry.field
        def example_decimal(self) -> Decimal:
            if False:
                print('Hello World!')
            return Decimal('3.14159')
    schema = strawberry.Schema(Query)
    result = schema.execute_sync('{ exampleDecimal }')
    assert not result.errors
    assert result.data['exampleDecimal'] == '3.14159'

def test_decimal_as_input():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def example_decimal(self, decimal: Decimal) -> Decimal:
            if False:
                for i in range(10):
                    print('nop')
            return decimal
    schema = strawberry.Schema(Query)
    result = schema.execute_sync('{ exampleDecimal(decimal: "3.14") }')
    assert not result.errors
    assert result.data['exampleDecimal'] == '3.14'

def test_serialization_of_incorrect_decimal_string():
    if False:
        print('Hello World!')
    '\n    Test GraphQLError is raised for an invalid Decimal.\n    The error should exclude "original_error".\n    '

    @strawberry.type
    class Query:
        ok: bool

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def decimal_input(self, decimal_input: Decimal) -> Decimal:
            if False:
                i = 10
                return i + 15
            return decimal_input
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    result = schema.execute_sync('\n            mutation decimalInput($value: Decimal!) {\n                decimalInput(decimalInput: $value)\n            }\n        ', variable_values={'value': 'fail'})
    assert result.errors
    assert isinstance(result.errors[0], GraphQLError)
    assert result.errors[0].original_error is None
    assert result.errors[0].message == 'Variable \'$value\' got invalid value \'fail\'; Value cannot represent a Decimal: "fail".'