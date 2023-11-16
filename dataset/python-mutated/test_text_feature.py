import pytest
from bigdl.dllib.feature.text import *
from test.bigdl.test_zoo_utils import ZooTestCase
text = 'Hello my friend, please annotate my text'

class TestTextFeature(ZooTestCase):

    def test_text_feature_with_label(self):
        if False:
            while True:
                i = 10
        feature = TextFeature(text, 1)
        assert feature.get_text() == text
        assert feature.get_label() == 1
        assert feature.has_label()
        assert set(feature.keys()) == {'text', 'label'}
        assert feature.get_tokens() is None
        assert feature.get_sample() is None

    def test_text_feature_without_label(self):
        if False:
            while True:
                i = 10
        feature = TextFeature(text)
        assert feature.get_text() == text
        assert feature.get_label() == -1
        assert not feature.has_label()
        assert feature.keys() == ['text']
        feature.set_label(0.0)
        assert feature.get_label() == 0
        assert feature.has_label()
        assert set(feature.keys()) == {'text', 'label'}
        assert feature.get_tokens() is None
        assert feature.get_sample() is None

    def test_text_feature_transformation(self):
        if False:
            while True:
                i = 10
        feature = TextFeature(text, 0)
        tokenizer = Tokenizer()
        tokenized = tokenizer.transform(feature)
        assert tokenized.get_tokens() == ['Hello', 'my', 'friend,', 'please', 'annotate', 'my', 'text']
        normalizer = Normalizer()
        normalized = normalizer.transform(tokenized)
        assert normalized.get_tokens() == ['hello', 'my', 'friend', 'please', 'annotate', 'my', 'text']
        word_index = {'my': 1, 'please': 2, 'friend': 3}
        indexed = WordIndexer(word_index).transform(normalized)
        shaped = SequenceShaper(5).transform(indexed)
        transformed = TextFeatureToSample().transform(shaped)
        assert set(transformed.keys()) == {'text', 'label', 'tokens', 'indexedTokens', 'sample'}
        sample = transformed.get_sample()
        assert list(sample.feature.storage) == [1.0, 3.0, 2.0, 1.0, 0.0]
        assert list(sample.label.storage) == [0.0]

    def test_text_feature_with_uri(self):
        if False:
            return 10
        feature = TextFeature(uri='A1')
        assert feature.get_text() is None
        assert feature.get_uri() == 'A1'
if __name__ == '__main__':
    pytest.main([__file__])