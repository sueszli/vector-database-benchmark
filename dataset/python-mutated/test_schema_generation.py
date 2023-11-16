from typing import Any, Dict, List, Optional
import pytest
from graphql import ExecutionContext as GraphQLExecutionContext
from graphql import ExecutionResult, GraphQLError, GraphQLField, GraphQLNonNull, GraphQLObjectType, GraphQLSchema, GraphQLString
from graphql import print_schema as graphql_core_print_schema
import strawberry

def test_generates_schema():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:
        example: str
    schema = strawberry.Schema(query=Query)
    target_schema = GraphQLSchema(query=GraphQLObjectType(name='Query', fields={'example': GraphQLField(GraphQLNonNull(GraphQLString), resolve=lambda obj, info: 'world')}))
    assert schema.as_str().strip() == graphql_core_print_schema(target_schema).strip()

def test_schema_introspect_returns_the_introspection_query_result():
    if False:
        return 10

    @strawberry.type
    class Query:
        example: str
    schema = strawberry.Schema(query=Query)
    introspection = schema.introspect()
    assert {'__schema'} == introspection.keys()
    assert {'queryType', 'mutationType', 'subscriptionType', 'types', 'directives'} == introspection['__schema'].keys()

def test_schema_fails_on_an_invalid_schema():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        ...
    with pytest.raises(ValueError, match='Invalid Schema. Errors.*'):
        strawberry.Schema(query=Query)

def test_custom_execution_context():
    if False:
        while True:
            i = 10

    class CustomExecutionContext(GraphQLExecutionContext):

        @staticmethod
        def build_response(data: Optional[Dict[str, Any]], errors: List[GraphQLError]) -> ExecutionResult:
            if False:
                i = 10
                return i + 15
            result = super(CustomExecutionContext, CustomExecutionContext).build_response(data, errors)
            if not result.data:
                return result
            result.data.update({'extra': 'data'})
            return result

    @strawberry.type
    class Query:
        hello: str = 'World'
    schema = strawberry.Schema(query=Query, execution_context_class=CustomExecutionContext)
    result = schema.execute_sync('{ hello }', root_value=Query())
    assert result.data == {'hello': 'World', 'extra': 'data'}