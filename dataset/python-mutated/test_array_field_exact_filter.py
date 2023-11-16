import pytest
from ...compat import ArrayField, MissingType

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_exact_no_match(schema):
    if False:
        return 10
    '\n    Test exact filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags: ["concert", "music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == []

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_exact_match(schema):
    if False:
        print('Hello World!')
    '\n    Test exact filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags: ["movie", "music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Musical'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_exact_empty_list(schema):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test exact filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags: []) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Speech'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_filter_schema_type(schema):
    if False:
        i = 10
        return i + 15
    '\n    Check that the type in the filter is an array field like on the object type.\n    '
    schema_str = str(schema)
    assert 'type EventType implements Node {\n  """The ID of the object"""\n  id: ID!\n  name: String!\n  tags: [String!]!\n  tagIds: [Int!]!\n  randomField: [Boolean!]!\n}' in schema_str
    filters = {'offset': 'Int', 'before': 'String', 'after': 'String', 'first': 'Int', 'last': 'Int', 'name': 'String', 'name_Contains': 'String', 'tags_Contains': '[String!]', 'tags_Overlap': '[String!]', 'tags': '[String!]', 'tags_Len': 'Int', 'tags_Len_In': '[Int]', 'tagsIds_Contains': '[Int!]', 'tagsIds_Overlap': '[Int!]', 'tagsIds': '[Int!]', 'randomField_Contains': '[Boolean!]', 'randomField_Overlap': '[Boolean!]', 'randomField': '[Boolean!]'}
    filters_str = ', '.join([f'{filter_field}: {gql_type}' for (filter_field, gql_type) in filters.items()])
    assert f'type Query {{\n  events({filters_str}): EventTypeConnection\n}}' in schema_str