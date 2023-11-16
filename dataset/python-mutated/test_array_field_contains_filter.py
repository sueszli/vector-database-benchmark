import pytest
from ...compat import ArrayField, MissingType

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_contains_multiple(schema):
    if False:
        return 10
    '\n    Test contains filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Contains: ["concert", "music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Live Show'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_contains_one(schema):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test contains filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Contains: ["music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Live Show'}}, {'node': {'name': 'Musical'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_contains_empty_list(schema):
    if False:
        while True:
            i = 10
    '\n    Test contains filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Contains: []) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Live Show'}}, {'node': {'name': 'Musical'}}, {'node': {'name': 'Ballet'}}, {'node': {'name': 'Speech'}}]