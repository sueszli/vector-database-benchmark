from unittest.mock import Mock
from graphql.error import GraphQLError
import strawberry
from strawberry.extensions import MaskErrors

def test_mask_all_errors():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def hidden_error(self) -> str:
            if False:
                i = 10
                return i + 15
            raise KeyError('This error is not visible')
    schema = strawberry.Schema(query=Query, extensions=[MaskErrors()])
    query = 'query { hiddenError }'
    result = schema.execute_sync(query)
    assert result.errors is not None
    formatted_errors = [err.formatted for err in result.errors]
    assert formatted_errors == [{'locations': [{'column': 9, 'line': 1}], 'message': 'Unexpected error.', 'path': ['hiddenError']}]

def test_mask_some_errors():
    if False:
        i = 10
        return i + 15

    class VisibleError(Exception):
        pass

    @strawberry.type
    class Query:

        @strawberry.field
        def visible_error(self) -> str:
            if False:
                i = 10
                return i + 15
            raise VisibleError('This error is visible')

        @strawberry.field
        def hidden_error(self) -> str:
            if False:
                return 10
            raise Exception('This error is not visible')

    def should_mask_error(error: GraphQLError) -> bool:
        if False:
            print('Hello World!')
        original_error = error.original_error
        if original_error and isinstance(original_error, VisibleError):
            return False
        return True
    schema = strawberry.Schema(query=Query, extensions=[MaskErrors(should_mask_error=should_mask_error)])
    query = 'query { hiddenError }'
    result = schema.execute_sync(query)
    assert result.errors is not None
    formatted_errors = [err.formatted for err in result.errors]
    assert formatted_errors == [{'locations': [{'column': 9, 'line': 1}], 'message': 'Unexpected error.', 'path': ['hiddenError']}]
    query = 'query { visibleError }'
    result = schema.execute_sync(query)
    assert result.errors is not None
    formatted_errors = [err.formatted for err in result.errors]
    assert formatted_errors == [{'locations': [{'column': 9, 'line': 1}], 'message': 'This error is visible', 'path': ['visibleError']}]

def test_process_errors_original_error():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:

        @strawberry.field
        def hidden_error(self) -> str:
            if False:
                return 10
            raise ValueError('This error is not visible')
    mock_process_error = Mock()

    class CustomSchema(strawberry.Schema):

        def process_errors(self, errors, execution_context):
            if False:
                for i in range(10):
                    print('nop')
            for error in errors:
                mock_process_error(error)
    schema = CustomSchema(query=Query, extensions=[MaskErrors()])
    query = 'query { hiddenError }'
    result = schema.execute_sync(query)
    assert result.errors is not None
    formatted_errors = [err.formatted for err in result.errors]
    assert formatted_errors == [{'locations': [{'column': 9, 'line': 1}], 'message': 'Unexpected error.', 'path': ['hiddenError']}]
    assert mock_process_error.call_count == 1
    call = mock_process_error.call_args_list[0]
    assert call[0][0].message == 'This error is not visible'
    assert isinstance(call[0][0].original_error, ValueError)

def test_graphql_error_masking():
    if False:
        return 10

    @strawberry.type
    class Query:

        @strawberry.field
        def graphql_error(self) -> str:
            if False:
                print('Hello World!')
            return None
    schema = strawberry.Schema(query=Query, extensions=[MaskErrors()])
    query = 'query { graphqlError }'
    result = schema.execute_sync(query)
    assert result.errors is not None
    formatted_errors = [err.formatted for err in result.errors]
    assert formatted_errors == [{'locations': [{'column': 9, 'line': 1}], 'message': 'Unexpected error.', 'path': ['graphqlError']}]