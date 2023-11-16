import logging
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import LabelField
from allennlp.data import Vocabulary

class TestLabelField(AllenNlpTestCase):

    def test_as_tensor_returns_integer_tensor(self):
        if False:
            i = 10
            return i + 15
        label = LabelField(5, skip_indexing=True)
        tensor = label.as_tensor(label.get_padding_lengths())
        assert tensor.item() == 5

    def test_label_field_can_index_with_vocab(self):
        if False:
            return 10
        vocab = Vocabulary()
        vocab.add_token_to_namespace('entailment', namespace='labels')
        vocab.add_token_to_namespace('contradiction', namespace='labels')
        vocab.add_token_to_namespace('neutral', namespace='labels')
        label = LabelField('entailment')
        label.index(vocab)
        tensor = label.as_tensor(label.get_padding_lengths())
        assert tensor.item() == 0

    def test_label_field_raises_with_non_integer_labels_and_no_indexing(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ConfigurationError):
            _ = LabelField('non integer field', skip_indexing=True)

    def test_label_field_raises_with_incorrect_label_type(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ConfigurationError):
            _ = LabelField([], skip_indexing=False)

    def test_label_field_empty_field_works(self):
        if False:
            print('Hello World!')
        label = LabelField('test')
        empty_label = label.empty_field()
        assert empty_label.label == -1

    def test_class_variables_for_namespace_warnings_work_correctly(self, caplog):
        if False:
            return 10
        with caplog.at_level(logging.WARNING, logger='allennlp.data.fields.label_field'):
            assert 'text' not in LabelField._already_warned_namespaces
            _ = LabelField('test', label_namespace='text')
            assert caplog.records
            assert 'text' in LabelField._already_warned_namespaces
            caplog.clear()
            _ = LabelField('test2', label_namespace='text')
            assert not caplog.records
            assert 'text2' not in LabelField._already_warned_namespaces
            caplog.clear()
            _ = LabelField('test', label_namespace='text2')
            assert caplog.records

    def test_printing_doesnt_crash(self):
        if False:
            i = 10
            return i + 15
        label = LabelField('label', label_namespace='namespace')
        print(label)

    def test_human_readable_dict(self):
        if False:
            while True:
                i = 10
        label = LabelField('apple', label_namespace='namespace')
        assert label.human_readable_repr() == 'apple'