from graphql import parse, validate
from ...types import Schema, ObjectType, String
from ..disable_introspection import DisableIntrospection

class Query(ObjectType):
    name = String(required=True)

    @staticmethod
    def resolve_name(root, info):
        if False:
            print('Hello World!')
        return 'Hello world!'
schema = Schema(query=Query)

def run_query(query: str):
    if False:
        for i in range(10):
            print('nop')
    document = parse(query)
    return validate(schema=schema.graphql_schema, document_ast=document, rules=(DisableIntrospection,))

def test_disallows_introspection_queries():
    if False:
        while True:
            i = 10
    errors = run_query('{ __schema { queryType { name } } }')
    assert len(errors) == 1
    assert errors[0].message == "Cannot query '__schema': introspection is disabled."

def test_allows_non_introspection_queries():
    if False:
        return 10
    errors = run_query('{ name }')
    assert len(errors) == 0