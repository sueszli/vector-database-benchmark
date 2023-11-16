from typing import Iterable
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import DatasetReader, InterleavingDatasetReader
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary

class PlainTextReader(DatasetReader):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = SpacyTokenizer()

    def _read(self, file_path: str) -> Iterable[Instance]:
        if False:
            for i in range(10):
                print('nop')
        with open(file_path) as input_file:
            for line in input_file:
                yield self.text_to_instance(line)

    def text_to_instance(self, line: str) -> Instance:
        if False:
            for i in range(10):
                print('nop')
        tokens = self._tokenizer.tokenize(line)
        return Instance({'line': TextField(tokens)})

    def apply_token_indexers(self, instance):
        if False:
            for i in range(10):
                print('nop')
        instance.fields['line'].token_indexers = self._token_indexers

class TestInterleavingDatasetReader(AllenNlpTestCase):

    def test_round_robin(self):
        if False:
            i = 10
            return i + 15
        readers = {'a': PlainTextReader(), 'b': PlainTextReader(), 'c': PlainTextReader()}
        reader = InterleavingDatasetReader(readers)
        data_dir = self.FIXTURES_ROOT / 'data'
        file_path = {'a': data_dir / 'babi.txt', 'b': data_dir / 'conll2003.txt', 'c': data_dir / 'conll2003.txt'}
        instances = list(reader.read(file_path))
        first_three_keys = {instance.fields['dataset'].metadata for instance in instances[:3]}
        assert first_three_keys == {'a', 'b', 'c'}
        next_three_keys = {instance.fields['dataset'].metadata for instance in instances[3:6]}
        assert next_three_keys == {'a', 'b', 'c'}

    def test_all_at_once(self):
        if False:
            return 10
        readers = {'f': PlainTextReader(), 'g': PlainTextReader(), 'h': PlainTextReader()}
        reader = InterleavingDatasetReader(readers, dataset_field_name='source', scheme='all_at_once')
        data_dir = self.FIXTURES_ROOT / 'data'
        file_path = f'''{{\n            "f": "{data_dir / 'babi.txt'}",\n            "g": "{data_dir / 'conll2003.txt'}",\n            "h": "{data_dir / 'conll2003.txt'}"\n        }}'''
        buckets = []
        last_source = None
        for instance in reader.read(file_path):
            source = instance.fields['source'].metadata
            if source != last_source:
                buckets.append([])
                last_source = source
            buckets[-1].append(instance)
        assert len(buckets) == 3

    @pytest.mark.parametrize('lazy', (True, False))
    def test_with_multi_process_loading(self, lazy):
        if False:
            while True:
                i = 10
        readers = {'a': PlainTextReader(), 'b': PlainTextReader(), 'c': PlainTextReader()}
        reader = InterleavingDatasetReader(readers)
        data_dir = self.FIXTURES_ROOT / 'data'
        file_path = {'a': data_dir / 'babi.txt', 'b': data_dir / 'conll2003.txt', 'c': data_dir / 'conll2003.txt'}
        vocab = Vocabulary.from_instances(reader.read(file_path))
        loader = MultiProcessDataLoader(reader, file_path, num_workers=1, batch_size=1, max_instances_in_memory=2 if lazy else None)
        loader.index_with(vocab)
        list(loader.iter_instances())
        list(loader)