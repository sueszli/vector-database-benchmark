import pytest
from ...compat import ArrayField, MissingType

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_overlap_multiple(schema):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test overlap filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Overlap: ["concert", "music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Live Show'}}, {'node': {'name': 'Musical'}}, {'node': {'name': 'Ballet'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_overlap_one(schema):
    if False:
        i = 10
        return i + 15
    '\n    Test overlap filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Overlap: ["music"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Live Show'}}, {'node': {'name': 'Musical'}}]

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_overlap_empty_list(schema):
    if False:
        i = 10
        return i + 15
    '\n    Test overlap filter on a array field of string.\n    '
    query = '\n    query {\n        events (tags_Overlap: []) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == []