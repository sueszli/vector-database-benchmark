import datetime
import dateutil.tz
import pytest
from graphql import GraphQLError
import strawberry
from strawberry.types.execution import ExecutionResult

@pytest.mark.parametrize(('typing', 'instance', 'serialized'), [(datetime.date, datetime.date(2019, 10, 25), '2019-10-25'), (datetime.datetime, datetime.datetime(2019, 10, 25, 13, 37), '2019-10-25T13:37:00'), (datetime.time, datetime.time(13, 37), '13:37:00')])
def test_serialization(typing, instance, serialized):
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:

        @strawberry.field
        def serialize(self) -> typing:
            if False:
                i = 10
                return i + 15
            return instance
    schema = strawberry.Schema(Query)
    result = schema.execute_sync('{ serialize }')
    assert not result.errors
    assert result.data['serialize'] == serialized

@pytest.mark.parametrize(('typing', 'name', 'instance', 'serialized'), [(datetime.date, 'Date', datetime.date(2019, 10, 25), '2019-10-25'), (datetime.datetime, 'DateTime', datetime.datetime(2019, 10, 25, 13, 37), '2019-10-25T13:37:00'), (datetime.datetime, 'DateTime', datetime.datetime(2019, 10, 25, 13, 37, tzinfo=dateutil.tz.tzutc()), '2019-10-25T13:37:00Z'), (datetime.time, 'Time', datetime.time(13, 37), '13:37:00')])
def test_deserialization(typing, name, instance, serialized):
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        deserialized = None

        @strawberry.field
        def deserialize(self, arg: typing) -> bool:
            if False:
                return 10
            Query.deserialized = arg
            return True
    schema = strawberry.Schema(Query)
    query = f'query Deserialize($value: {name}!) {{\n        deserialize(arg: $value)\n    }}'
    result = schema.execute_sync(query, variable_values={'value': serialized})
    assert not result.errors
    assert Query.deserialized == instance

@pytest.mark.parametrize(('typing', 'instance', 'serialized'), [(datetime.date, datetime.date(2019, 10, 25), '2019-10-25'), (datetime.datetime, datetime.datetime(2019, 10, 25, 13, 37), '2019-10-25T13:37:00'), (datetime.time, datetime.time(13, 37), '13:37:00')])
def test_deserialization_with_parse_literal(typing, instance, serialized):
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:
        deserialized = None

        @strawberry.field
        def deserialize(self, arg: typing) -> bool:
            if False:
                return 10
            Query.deserialized = arg
            return True
    schema = strawberry.Schema(Query)
    query = f'query Deserialize {{\n        deserialize(arg: "{serialized}")\n    }}'
    result = schema.execute_sync(query)
    assert not result.errors
    assert Query.deserialized == instance

def execute_mutation(value) -> ExecutionResult:
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        ok: bool

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def datetime_input(self, datetime_input: datetime.datetime) -> datetime.datetime:
            if False:
                i = 10
                return i + 15
            assert isinstance(datetime_input, datetime.datetime)
            return datetime_input
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    return schema.execute_sync('\n            mutation datetimeInput($value: DateTime!) {\n                datetimeInput(datetimeInput: $value)\n            }\n        ', variable_values={'value': value})

@pytest.mark.parametrize('value', ('2012-13-01', '2012-04-9', '20120411T03:30+', '20120411T03:30+1234567', '20120411T03:30-25:40', '20120411T03:30+00:60', '20120411T03:30+00:61', '20120411T033030.123456012:002014-03-12T12:30:14', '2014-04-21T24:00:01'))
def test_serialization_of_incorrect_datetime_string(value):
    if False:
        return 10
    '\n    Test GraphQLError is raised for incorrect datetime.\n    The error should exclude "original_error".\n    '
    result = execute_mutation(value)
    assert result.errors
    assert isinstance(result.errors[0], GraphQLError)
    assert result.errors[0].original_error is None

def test_serialization_error_message_for_incorrect_datetime_string():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if error message is using original error message\n    from datetime lib, and is properly formatted\n    '
    result = execute_mutation('2021-13-01T09:00:00')
    assert result.errors
    assert result.errors[0].message == 'Variable \'$value\' got invalid value \'2021-13-01T09:00:00\'; Value cannot represent a DateTime: "2021-13-01T09:00:00". month must be in 1..12'