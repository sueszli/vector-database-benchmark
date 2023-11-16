from unittest.mock import patch
import pytest
from graphql import parse
import strawberry
from strawberry.extensions import MaxTokensLimiter, ParserCache

@patch('strawberry.schema.execute.parse', wraps=parse)
def test_parser_cache_extension(mock_parse):
    if False:
        return 10

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self) -> str:
            if False:
                return 10
            return 'world'

        @strawberry.field
        def ping(self) -> str:
            if False:
                print('Hello World!')
            return 'pong'
    schema = strawberry.Schema(query=Query, extensions=[ParserCache()])
    query = 'query { hello }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'hello': 'world'}
    assert mock_parse.call_count == 1
    for _ in range(3):
        result = schema.execute_sync(query)
        assert not result.errors
        assert result.data == {'hello': 'world'}
    assert mock_parse.call_count == 1
    query2 = 'query { ping }'
    result = schema.execute_sync(query2)
    assert not result.errors
    assert result.data == {'ping': 'pong'}
    assert mock_parse.call_count == 2

@patch('strawberry.schema.execute.parse', wraps=parse)
def test_parser_cache_extension_arguments(mock_parse):
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'world'

        @strawberry.field
        def ping(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'pong'
    schema = strawberry.Schema(query=Query, extensions=[MaxTokensLimiter(max_token_count=20), ParserCache()])
    query = 'query { hello }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'hello': 'world'}
    mock_parse.assert_called_with('query { hello }', max_tokens=20)

@patch('strawberry.schema.execute.parse', wraps=parse)
def test_parser_cache_extension_max_size(mock_parse):
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self) -> str:
            if False:
                while True:
                    i = 10
            return 'world'

        @strawberry.field
        def ping(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'pong'
    schema = strawberry.Schema(query=Query, extensions=[ParserCache(maxsize=1)])
    query = 'query { hello }'
    for _ in range(2):
        result = schema.execute_sync(query)
        assert not result.errors
        assert result.data == {'hello': 'world'}
    assert mock_parse.call_count == 1
    query2 = 'query { ping }'
    result = schema.execute_sync(query2)
    assert not result.errors
    assert mock_parse.call_count == 2
    result = schema.execute_sync(query)
    assert not result.errors
    assert mock_parse.call_count == 3

@pytest.mark.asyncio
async def test_parser_cache_extension_async():

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'world'

        @strawberry.field
        def ping(self) -> str:
            if False:
                print('Hello World!')
            return 'pong'
    schema = strawberry.Schema(query=Query, extensions=[ParserCache()])
    query = 'query { hello }'
    with patch('strawberry.schema.execute.parse', wraps=parse) as mock_parse:
        result = await schema.execute(query)
        assert not result.errors
        assert result.data == {'hello': 'world'}
        assert mock_parse.call_count == 1
        for _ in range(3):
            result = await schema.execute(query)
            assert not result.errors
            assert result.data == {'hello': 'world'}
        assert mock_parse.call_count == 1
        query2 = 'query { ping }'
        result = await schema.execute(query2)
        assert not result.errors
        assert result.data == {'ping': 'pong'}
        assert mock_parse.call_count == 2