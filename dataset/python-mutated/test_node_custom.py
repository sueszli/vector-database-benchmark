from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node

class CustomNode(Node):

    class Meta:
        name = 'Node'

    @staticmethod
    def to_global_id(type_, id):
        if False:
            i = 10
            return i + 15
        return id

    @staticmethod
    def get_node_from_global_id(info, id, only_type=None):
        if False:
            while True:
                i = 10
        assert info.schema is graphql_schema
        if id in user_data:
            return user_data.get(id)
        else:
            return photo_data.get(id)

class BasePhoto(Interface):
    width = Int(description='The width of the photo in pixels')

class User(ObjectType):

    class Meta:
        interfaces = [CustomNode]
    name = String(description='The full name of the user')

class Photo(ObjectType):

    class Meta:
        interfaces = [CustomNode, BasePhoto]
user_data = {'1': User(id='1', name='John Doe'), '2': User(id='2', name='Jane Smith')}
photo_data = {'3': Photo(id='3', width=300), '4': Photo(id='4', width=400)}

class RootQuery(ObjectType):
    node = CustomNode.Field()
schema = Schema(query=RootQuery, types=[User, Photo])
graphql_schema = schema.graphql_schema

def test_str_schema_correct():
    if False:
        return 10
    assert str(schema).strip() == dedent('\n        schema {\n          query: RootQuery\n        }\n\n        type User implements Node {\n          """The ID of the object"""\n          id: ID!\n\n          """The full name of the user"""\n          name: String\n        }\n\n        interface Node {\n          """The ID of the object"""\n          id: ID!\n        }\n\n        type Photo implements Node & BasePhoto {\n          """The ID of the object"""\n          id: ID!\n\n          """The width of the photo in pixels"""\n          width: Int\n        }\n\n        interface BasePhoto {\n          """The width of the photo in pixels"""\n          width: Int\n        }\n\n        type RootQuery {\n          node(\n            """The ID of the object"""\n            id: ID!\n          ): Node\n        }\n        ').strip()

def test_gets_the_correct_id_for_users():
    if False:
        print('Hello World!')
    query = '\n      {\n        node(id: "1") {\n          id\n        }\n      }\n    '
    expected = {'node': {'id': '1'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_gets_the_correct_id_for_photos():
    if False:
        while True:
            i = 10
    query = '\n      {\n        node(id: "4") {\n          id\n        }\n      }\n    '
    expected = {'node': {'id': '4'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_gets_the_correct_name_for_users():
    if False:
        i = 10
        return i + 15
    query = '\n      {\n        node(id: "1") {\n          id\n          ... on User {\n            name\n          }\n        }\n      }\n    '
    expected = {'node': {'id': '1', 'name': 'John Doe'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_gets_the_correct_width_for_photos():
    if False:
        while True:
            i = 10
    query = '\n      {\n        node(id: "4") {\n          id\n          ... on Photo {\n            width\n          }\n        }\n      }\n    '
    expected = {'node': {'id': '4', 'width': 400}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_gets_the_correct_typename_for_users():
    if False:
        for i in range(10):
            print('nop')
    query = '\n      {\n        node(id: "1") {\n          id\n          __typename\n        }\n      }\n    '
    expected = {'node': {'id': '1', '__typename': 'User'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_gets_the_correct_typename_for_photos():
    if False:
        print('Hello World!')
    query = '\n      {\n        node(id: "4") {\n          id\n          __typename\n        }\n      }\n    '
    expected = {'node': {'id': '4', '__typename': 'Photo'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_ignores_photo_fragments_on_user():
    if False:
        i = 10
        return i + 15
    query = '\n      {\n        node(id: "1") {\n          id\n          ... on Photo {\n            width\n          }\n        }\n      }\n    '
    expected = {'node': {'id': '1'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_returns_null_for_bad_ids():
    if False:
        print('Hello World!')
    query = '\n      {\n        node(id: "5") {\n          id\n        }\n      }\n    '
    expected = {'node': None}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_have_correct_node_interface():
    if False:
        while True:
            i = 10
    query = '\n      {\n        __type(name: "Node") {\n          name\n          kind\n          fields {\n            name\n            type {\n              kind\n              ofType {\n                name\n                kind\n              }\n            }\n          }\n        }\n      }\n    '
    expected = {'__type': {'name': 'Node', 'kind': 'INTERFACE', 'fields': [{'name': 'id', 'type': {'kind': 'NON_NULL', 'ofType': {'name': 'ID', 'kind': 'SCALAR'}}}]}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected

def test_has_correct_node_root_field():
    if False:
        while True:
            i = 10
    query = '\n      {\n        __schema {\n          queryType {\n            fields {\n              name\n              type {\n                name\n                kind\n              }\n              args {\n                name\n                type {\n                  kind\n                  ofType {\n                    name\n                    kind\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n    '
    expected = {'__schema': {'queryType': {'fields': [{'name': 'node', 'type': {'name': 'Node', 'kind': 'INTERFACE'}, 'args': [{'name': 'id', 'type': {'kind': 'NON_NULL', 'ofType': {'name': 'ID', 'kind': 'SCALAR'}}}]}]}}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected