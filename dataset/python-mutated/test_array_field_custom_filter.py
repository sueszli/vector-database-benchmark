import pytest
from ...compat import ArrayField, MissingType

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_len_filter(schema):
    if False:
        while True:
            i = 10
    query = '\n    query {\n        events (tags_Len: 2) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Musical'}}, {'node': {'name': 'Ballet'}}]
    query = '\n    query {\n        events (tags_Len: 0) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Speech'}}]
    query = '\n    query {\n        events (tags_Len: 10) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == []
    query = '\n    query {\n        events (tags_Len: "2") {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert len(result.errors) == 1
    assert result.errors[0].message == 'Int cannot represent non-integer value: "2"'
    query = '\n    query {\n        events (tags_Len: True) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert len(result.errors) == 1
    assert result.errors[0].message == 'Int cannot represent non-integer value: True'

@pytest.mark.skipif(ArrayField is MissingType, reason='ArrayField should exist')
def test_array_field_custom_filter(schema):
    if False:
        return 10
    query = '\n    query {\n        events (tags_Len_In: 2) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Ballet'}}, {'node': {'name': 'Musical'}}]
    query = '\n    query {\n        events (tags_Len_In: [0, 2]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == [{'node': {'name': 'Ballet'}}, {'node': {'name': 'Musical'}}, {'node': {'name': 'Speech'}}]
    query = '\n    query {\n        events (tags_Len_In: [10]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == []
    query = '\n    query {\n        events (tags_Len_In: []) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['events']['edges'] == []
    query = '\n    query {\n        events (tags_Len_In: "12") {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert len(result.errors) == 1
    assert result.errors[0].message == 'Int cannot represent non-integer value: "12"'
    query = '\n    query {\n        events (tags_Len_In: True) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert len(result.errors) == 1
    assert result.errors[0].message == 'Int cannot represent non-integer value: True'