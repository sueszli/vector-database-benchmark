import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from textwrap import dedent
from typing import Optional
from uuid import UUID
import pytest
import strawberry
from strawberry import scalar
from strawberry.exceptions import ScalarAlreadyRegisteredError
from strawberry.scalars import JSON, Base16, Base32, Base64
from strawberry.schema.types.base_scalars import Date

def test_void_function():
    if False:
        i = 10
        return i + 15
    NoneType = type(None)

    @strawberry.type
    class Query:

        @strawberry.field
        def void_ret(self) -> None:
            if False:
                return 10
            return

        @strawberry.field
        def void_ret_crash(self) -> NoneType:
            if False:
                for i in range(10):
                    print('nop')
            return 1

        @strawberry.field
        def void_arg(self, x: None) -> None:
            if False:
                for i in range(10):
                    print('nop')
            return
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('\n      type Query {\n        voidRet: Void\n        voidRetCrash: Void\n        voidArg(x: Void): Void\n      }\n\n      """Represents NULL values"""\n      scalar Void\n    ').strip()
    result = schema.execute_sync('query { voidRet }')
    assert not result.errors
    assert result.data == {'voidRet': None}
    result = schema.execute_sync('query { voidArg (x: null) }')
    assert not result.errors
    assert result.data == {'voidArg': None}
    result = schema.execute_sync('query { voidArg (x: 1) }')
    assert result.errors
    result = schema.execute_sync('query { voidRetCrash }')
    assert result.errors

def test_uuid_field_string_value():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:
        unique_id: UUID
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('\n      type Query {\n        uniqueId: UUID!\n      }\n\n      scalar UUID\n    ').strip()
    result = schema.execute_sync('query { uniqueId }', root_value=Query(unique_id='e350746c-33b6-4469-86b0-5f16e1e12232'))
    assert not result.errors
    assert result.data == {'uniqueId': 'e350746c-33b6-4469-86b0-5f16e1e12232'}

def test_uuid_field_uuid_value():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        unique_id: UUID
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('\n      type Query {\n        uniqueId: UUID!\n      }\n\n      scalar UUID\n    ').strip()
    result = schema.execute_sync('query { uniqueId }', root_value=Query(unique_id=UUID('e350746c-33b6-4469-86b0-5f16e1e12232')))
    assert not result.errors
    assert result.data == {'uniqueId': 'e350746c-33b6-4469-86b0-5f16e1e12232'}

def test_uuid_input():
    if False:
        return 10

    @strawberry.type
    class Query:
        ok: bool

    @strawberry.type
    class Mutation:

        @strawberry.mutation
        def uuid_input(self, input_id: UUID) -> str:
            if False:
                return 10
            assert isinstance(input_id, UUID)
            return str(input_id)
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    result = schema.execute_sync('\n        mutation {\n            uuidInput(inputId: "e350746c-33b6-4469-86b0-5f16e1e12232")\n        }\n    ')
    assert not result.errors
    assert result.data == {'uuidInput': 'e350746c-33b6-4469-86b0-5f16e1e12232'}

def test_json():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:

        @strawberry.field
        def echo_json(data: JSON) -> JSON:
            if False:
                while True:
                    i = 10
            return data

        @strawberry.field
        def echo_json_nullable(data: Optional[JSON]) -> Optional[JSON]:
            if False:
                return 10
            return data
    schema = strawberry.Schema(query=Query)
    expected_schema = dedent('\n        """\n        The `JSON` scalar type represents JSON values as specified by [ECMA-404](http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf).\n        """\n        scalar JSON @specifiedBy(url: "http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf")\n\n        type Query {\n          echoJson(data: JSON!): JSON!\n          echoJsonNullable(data: JSON): JSON\n        }\n        ').strip()
    assert str(schema) == expected_schema
    result = schema.execute_sync('\n        query {\n            echoJson(data: {hello: {a: 1}, someNumbers: [1, 2, 3], null: null})\n            echoJsonNullable(data: {hello: {a: 1}, someNumbers: [1, 2, 3], null: null})\n        }\n    ')
    assert not result.errors
    assert result.data == {'echoJson': {'hello': {'a': 1}, 'someNumbers': [1, 2, 3], 'null': None}, 'echoJsonNullable': {'hello': {'a': 1}, 'someNumbers': [1, 2, 3], 'null': None}}
    result = schema.execute_sync('\n        query {\n            echoJson(data: null)\n        }\n    ')
    assert result.errors
    result = schema.execute_sync('\n        query {\n            echoJsonNullable(data: null)\n        }\n    ')
    assert not result.errors
    assert result.data == {'echoJsonNullable': None}

def test_base16():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:

        @strawberry.field
        def base16_encode(data: str) -> Base16:
            if False:
                print('Hello World!')
            return bytes(data, 'utf-8')

        @strawberry.field
        def base16_decode(data: Base16) -> str:
            if False:
                while True:
                    i = 10
            return data.decode('utf-8')

        @strawberry.field
        def base32_encode(data: str) -> Base32:
            if False:
                i = 10
                return i + 15
            return bytes(data, 'utf-8')

        @strawberry.field
        def base32_decode(data: Base32) -> str:
            if False:
                i = 10
                return i + 15
            return data.decode('utf-8')

        @strawberry.field
        def base64_encode(data: str) -> Base64:
            if False:
                print('Hello World!')
            return bytes(data, 'utf-8')

        @strawberry.field
        def base64_decode(data: Base64) -> str:
            if False:
                i = 10
                return i + 15
            return data.decode('utf-8')
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('\n        """Represents binary data as Base16-encoded (hexadecimal) strings."""\n        scalar Base16 @specifiedBy(url: "https://datatracker.ietf.org/doc/html/rfc4648.html#section-8")\n\n        """\n        Represents binary data as Base32-encoded strings, using the standard alphabet.\n        """\n        scalar Base32 @specifiedBy(url: "https://datatracker.ietf.org/doc/html/rfc4648.html#section-6")\n\n        """\n        Represents binary data as Base64-encoded strings, using the standard alphabet.\n        """\n        scalar Base64 @specifiedBy(url: "https://datatracker.ietf.org/doc/html/rfc4648.html#section-4")\n\n        type Query {\n          base16Encode(data: String!): Base16!\n          base16Decode(data: Base16!): String!\n          base32Encode(data: String!): Base32!\n          base32Decode(data: Base32!): String!\n          base64Encode(data: String!): Base64!\n          base64Decode(data: Base64!): String!\n        }\n    ').strip()
    result = schema.execute_sync('\n        query {\n            base16Encode(data: "Hello")\n            base16Decode(data: "48656c6C6f")  # < Mix lowercase and uppercase\n            base32Encode(data: "Hello")\n            base32Decode(data: "JBSWY3dp")  # < Mix lowercase and uppercase\n            base64Encode(data: "Hello")\n            base64Decode(data: "SGVsbG8=")\n        }\n    ')
    assert not result.errors
    assert result.data == {'base16Encode': '48656C6C6F', 'base16Decode': 'Hello', 'base32Encode': 'JBSWY3DP', 'base32Decode': 'Hello', 'base64Encode': 'SGVsbG8=', 'base64Decode': 'Hello'}

def test_override_built_in_scalars():
    if False:
        while True:
            i = 10
    EpochDateTime = strawberry.scalar(datetime, serialize=lambda value: int(value.timestamp()), parse_value=lambda value: datetime.fromtimestamp(int(value), timezone.utc))

    @strawberry.type
    class Query:

        @strawberry.field
        def current_time(self) -> datetime:
            if False:
                while True:
                    i = 10
            return datetime(2021, 8, 11, 12, 0, tzinfo=timezone.utc)

        @strawberry.field
        def isoformat(self, input_datetime: datetime) -> str:
            if False:
                i = 10
                return i + 15
            return input_datetime.isoformat()
    schema = strawberry.Schema(Query, scalar_overrides={datetime: EpochDateTime})
    result = schema.execute_sync('\n        {\n            currentTime\n            isoformat(inputDatetime: 1628683200)\n        }\n        ')
    assert not result.errors
    assert result.data['currentTime'] == 1628683200
    assert result.data['isoformat'] == '2021-08-11T12:00:00+00:00'

def test_override_unknown_scalars():
    if False:
        return 10
    Duration = strawberry.scalar(timedelta, name='Duration', serialize=timedelta.total_seconds, parse_value=lambda s: timedelta(seconds=s))

    @strawberry.type
    class Query:

        @strawberry.field
        def duration(self, value: timedelta) -> timedelta:
            if False:
                while True:
                    i = 10
            return value
    schema = strawberry.Schema(Query, scalar_overrides={timedelta: Duration})
    result = schema.execute_sync('{ duration(value: 10) }')
    assert not result.errors
    assert result.data == {'duration': 10}

def test_decimal():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def decimal(value: Decimal) -> Decimal:
            if False:
                print('Hello World!')
            return value
    schema = strawberry.Schema(query=Query)
    result = schema.execute_sync('\n        query {\n            floatDecimal: decimal(value: 3.14)\n            floatDecimal2: decimal(value: 3.14509999)\n            floatDecimal3: decimal(value: 0.000001)\n            stringDecimal: decimal(value: "3.14")\n            stringDecimal2: decimal(value: "3.1499999991")\n        }\n    ')
    assert not result.errors
    assert result.data == {'floatDecimal': '3.14', 'floatDecimal2': '3.14509999', 'floatDecimal3': '0.000001', 'stringDecimal': '3.14', 'stringDecimal2': '3.1499999991'}

@pytest.mark.raises_strawberry_exception(ScalarAlreadyRegisteredError, match='Scalar `MyCustomScalar` has already been registered')
def test_duplicate_scalars_raises_exception():
    if False:
        print('Hello World!')
    MyCustomScalar = strawberry.scalar(str, name='MyCustomScalar')
    MyCustomScalar2 = strawberry.scalar(int, name='MyCustomScalar')

    @strawberry.type
    class Query:
        scalar_1: MyCustomScalar
        scalar_2: MyCustomScalar2
    strawberry.Schema(Query)

@pytest.mark.raises_strawberry_exception(ScalarAlreadyRegisteredError, match='Scalar `MyCustomScalar` has already been registered')
def test_duplicate_scalars_raises_exception_using_alias():
    if False:
        for i in range(10):
            print('nop')
    MyCustomScalar = scalar(str, name='MyCustomScalar')
    MyCustomScalar2 = scalar(int, name='MyCustomScalar')

    @strawberry.type
    class Query:
        scalar_1: MyCustomScalar
        scalar_2: MyCustomScalar2
    strawberry.Schema(Query)

@pytest.mark.skipif(sys.version_info < (3, 10), reason='pipe syntax for union is only available on python 3.10+')
def test_optional_scalar_with_or_operator():
    if False:
        return 10
    'Check `|` operator support with an optional scalar.'

    @strawberry.type
    class Query:
        date: Date | None
    schema = strawberry.Schema(query=Query)
    query = '{ date }'
    result = schema.execute_sync(query, root_value=Query(date=None))
    assert not result.errors
    assert result.data['date'] is None
    result = schema.execute_sync(query, root_value=Query(date=date(2020, 1, 1)))
    assert not result.errors
    assert result.data['date'] == '2020-01-01'