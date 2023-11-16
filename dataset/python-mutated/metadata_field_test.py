import pytest
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data.fields import MetadataField

class TestMetadataField(AllenNlpTestCase):

    def test_mapping_works_with_dict(self):
        if False:
            i = 10
            return i + 15
        field = MetadataField({'a': 1, 'b': [0]})
        assert 'a' in field
        assert field['a'] == 1
        assert len(field) == 2
        keys = {k for k in field}
        assert keys == {'a', 'b'}
        values = [v for v in field.values()]
        assert len(values) == 2
        assert 1 in values
        assert [0] in values

    def test_mapping_raises_with_non_dict(self):
        if False:
            i = 10
            return i + 15
        field = MetadataField(0)
        with pytest.raises(TypeError):
            _ = field[0]
        with pytest.raises(TypeError):
            _ = len(field)
        with pytest.raises(TypeError):
            _ = [x for x in field]

    def test_human_readable_repr(self):
        if False:
            for i in range(10):
                print('nop')
        field = MetadataField({'a': 1, 'b': [0]})
        assert field.human_readable_repr() == {'a': 1, 'b': [0]}