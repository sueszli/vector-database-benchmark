import codecs
import gzip
import pickle
import shutil
import zipfile
from copy import deepcopy
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import _NamespaceDependentDefaultDict, _read_pretrained_tokens, DEFAULT_OOV_TOKEN, Vocabulary
from allennlp.modules.token_embedders.embedding import format_embeddings_file_uri

class TestVocabulary(AllenNlpTestCase):

    def setup_method(self):
        if False:
            print('Hello World!')
        token_indexer = SingleIdTokenIndexer('tokens')
        text_field = TextField([Token(t) for t in ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c']], {'tokens': token_indexer})
        self.instance = Instance({'text': text_field})
        self.dataset = Batch([self.instance])
        super().setup_method()

    def test_pickling(self):
        if False:
            i = 10
            return i + 15
        vocab = Vocabulary.from_instances(self.dataset)
        pickled = pickle.dumps(vocab)
        unpickled = pickle.loads(pickled)
        assert dict(unpickled._index_to_token) == dict(vocab._index_to_token)
        assert dict(unpickled._token_to_index) == dict(vocab._token_to_index)
        assert unpickled._non_padded_namespaces == vocab._non_padded_namespaces
        assert unpickled._oov_token == vocab._oov_token
        assert unpickled._padding_token == vocab._padding_token
        assert unpickled._retained_counter == vocab._retained_counter

    def test_from_dataset_respects_max_vocab_size_single_int(self):
        if False:
            print('Hello World!')
        max_vocab_size = 1
        vocab = Vocabulary.from_instances(self.dataset, max_vocab_size=max_vocab_size)
        words = vocab.get_index_to_token_vocabulary().values()
        assert len(words) == max_vocab_size + 2
        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert len(words) == 5

    def test_from_dataset_respects_min_count(self):
        if False:
            print('Hello World!')
        vocab = Vocabulary.from_instances(self.dataset, min_count={'tokens': 4})
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words
        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' in words

    def test_from_dataset_respects_exclusive_embedding_file(self):
        if False:
            print('Hello World!')
        embeddings_filename = str(self.TEST_DIR / 'embeddings.gz')
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write('a 1.0 2.3 -1.0\n'.encode('utf-8'))
            embeddings_file.write('b 0.1 0.4 -4.0\n'.encode('utf-8'))
        vocab = Vocabulary.from_instances(self.dataset, min_count={'tokens': 4}, pretrained_files={'tokens': embeddings_filename}, only_include_pretrained_words=True)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words
        vocab = Vocabulary.from_instances(self.dataset, pretrained_files={'tokens': embeddings_filename}, only_include_pretrained_words=True)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' not in words

    def test_from_dataset_respects_inclusive_embedding_file(self):
        if False:
            print('Hello World!')
        embeddings_filename = str(self.TEST_DIR / 'embeddings.gz')
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write('a 1.0 2.3 -1.0\n'.encode('utf-8'))
            embeddings_file.write('b 0.1 0.4 -4.0\n'.encode('utf-8'))
        vocab = Vocabulary.from_instances(self.dataset, min_count={'tokens': 4}, pretrained_files={'tokens': embeddings_filename}, only_include_pretrained_words=False)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' not in words
        vocab = Vocabulary.from_instances(self.dataset, pretrained_files={'tokens': embeddings_filename}, only_include_pretrained_words=False)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' in words

    def test_add_word_to_index_gives_consistent_results(self):
        if False:
            return 10
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace('word')
        assert 'word' in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index('word') == word_index
        assert vocab.get_token_from_index(word_index) == 'word'
        assert vocab.get_vocab_size() == initial_vocab_size + 1
        vocab.add_token_to_namespace('word')
        assert 'word' in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index('word') == word_index
        assert vocab.get_token_from_index(word_index) == 'word'
        assert vocab.get_vocab_size() == initial_vocab_size + 1

    def test_namespaces(self):
        if False:
            return 10
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace('word', namespace='1')
        assert 'word' in vocab.get_index_to_token_vocabulary(namespace='1').values()
        assert vocab.get_token_index('word', namespace='1') == word_index
        assert vocab.get_token_from_index(word_index, namespace='1') == 'word'
        assert vocab.get_vocab_size(namespace='1') == initial_vocab_size + 1
        word2_index = vocab.add_token_to_namespace('word2', namespace='2')
        word_index = vocab.add_token_to_namespace('word', namespace='2')
        assert 'word' in vocab.get_index_to_token_vocabulary(namespace='2').values()
        assert 'word2' in vocab.get_index_to_token_vocabulary(namespace='2').values()
        assert vocab.get_token_index('word', namespace='2') == word_index
        assert vocab.get_token_index('word2', namespace='2') == word2_index
        assert vocab.get_token_from_index(word_index, namespace='2') == 'word'
        assert vocab.get_token_from_index(word2_index, namespace='2') == 'word2'
        assert vocab.get_vocab_size(namespace='2') == initial_vocab_size + 2

    def test_namespace_dependent_default_dict(self):
        if False:
            while True:
                i = 10
        default_dict = _NamespaceDependentDefaultDict(['bar', '*baz'], lambda : 7, lambda : 3)
        assert default_dict['foo'] == 7
        assert default_dict['baz'] == 3
        assert default_dict['bar'] == 3
        assert default_dict['foobaz'] == 3

    def test_unknown_token(self):
        if False:
            print('Hello World!')
        vocab = Vocabulary()
        oov_token = vocab._oov_token
        oov_index = vocab.get_token_index(oov_token)
        assert oov_index == 1
        assert vocab.get_token_index('unseen word') == oov_index

    def test_get_token_index(self):
        if False:
            i = 10
            return i + 15
        vocab = Vocabulary(counter={'labels': {'foo': 3, 'bar': 2}, 'tokens': {'foo': 3, 'bar': 2}}, non_padded_namespaces=['labels'])
        expected_token_to_index_dicts = {'tokens': {vocab._padding_token: 0, vocab._oov_token: 1, 'foo': 2, 'bar': 3}, 'labels': {'foo': 0, 'bar': 1}}
        assert vocab._token_to_index['tokens'] == expected_token_to_index_dicts['tokens']
        assert vocab._token_to_index['labels'] == expected_token_to_index_dicts['labels']
        assert vocab.get_token_index('baz', 'tokens') == 1
        with pytest.raises(KeyError, match="'baz' not found .* and namespace does not contain the default OOV token .*"):
            vocab.get_token_index('baz', 'labels')
        with pytest.raises(KeyError, match=f"'{vocab._oov_token}' not found .*"):
            vocab.get_token_index(vocab._oov_token, 'labels')
        assert vocab._token_to_index['tokens'] == expected_token_to_index_dicts['tokens']
        assert vocab._token_to_index['labels'] == expected_token_to_index_dicts['labels']

    def test_set_from_file_reads_padded_files(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_filename = self.TEST_DIR / 'vocab_file'
        with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
            vocab_file.write('<S>\n')
            vocab_file.write('</S>\n')
            vocab_file.write('<UNK>\n')
            vocab_file.write('a\n')
            vocab_file.write('tricky\x0bchar\n')
            vocab_file.write('word\n')
            vocab_file.write('another\n')
        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=True, oov_token='<UNK>')
        assert vocab._oov_token == DEFAULT_OOV_TOKEN
        assert vocab.get_token_index('random string') == 3
        assert vocab.get_token_index('<S>') == 1
        assert vocab.get_token_index('</S>') == 2
        assert vocab.get_token_index(DEFAULT_OOV_TOKEN) == 3
        assert vocab.get_token_index('a') == 4
        assert vocab.get_token_index('tricky\x0bchar') == 5
        assert vocab.get_token_index('word') == 6
        assert vocab.get_token_index('another') == 7
        assert vocab.get_token_from_index(0) == vocab._padding_token
        assert vocab.get_token_from_index(1) == '<S>'
        assert vocab.get_token_from_index(2) == '</S>'
        assert vocab.get_token_from_index(3) == DEFAULT_OOV_TOKEN
        assert vocab.get_token_from_index(4) == 'a'
        assert vocab.get_token_from_index(5) == 'tricky\x0bchar'
        assert vocab.get_token_from_index(6) == 'word'
        assert vocab.get_token_from_index(7) == 'another'

    def test_set_from_file_reads_non_padded_files(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_filename = self.TEST_DIR / 'vocab_file'
        with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
            vocab_file.write('B-PERS\n')
            vocab_file.write('I-PERS\n')
            vocab_file.write('O\n')
            vocab_file.write('B-ORG\n')
            vocab_file.write('I-ORG\n')
        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=False, namespace='tags')
        assert vocab.get_token_index('B-PERS', namespace='tags') == 0
        assert vocab.get_token_index('I-PERS', namespace='tags') == 1
        assert vocab.get_token_index('O', namespace='tags') == 2
        assert vocab.get_token_index('B-ORG', namespace='tags') == 3
        assert vocab.get_token_index('I-ORG', namespace='tags') == 4
        assert vocab.get_token_from_index(0, namespace='tags') == 'B-PERS'
        assert vocab.get_token_from_index(1, namespace='tags') == 'I-PERS'
        assert vocab.get_token_from_index(2, namespace='tags') == 'O'
        assert vocab.get_token_from_index(3, namespace='tags') == 'B-ORG'
        assert vocab.get_token_from_index(4, namespace='tags') == 'I-ORG'

    def test_saving_and_loading(self):
        if False:
            return 10
        vocab_dir = self.TEST_DIR / 'vocab_save'
        vocab = Vocabulary(non_padded_namespaces=['a', 'c'])
        vocab.add_tokens_to_namespace(['a0', 'a1', 'a2'], namespace='a')
        vocab.add_tokens_to_namespace(['b2', 'b3'], namespace='b')
        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)
        assert vocab2._non_padded_namespaces == {'a', 'c'}
        assert vocab2.get_vocab_size(namespace='a') == 3
        assert vocab2.get_token_from_index(0, namespace='a') == 'a0'
        assert vocab2.get_token_from_index(1, namespace='a') == 'a1'
        assert vocab2.get_token_from_index(2, namespace='a') == 'a2'
        assert vocab2.get_token_index('a0', namespace='a') == 0
        assert vocab2.get_token_index('a1', namespace='a') == 1
        assert vocab2.get_token_index('a2', namespace='a') == 2
        assert vocab2.get_vocab_size(namespace='b') == 4
        assert vocab2.get_token_from_index(0, namespace='b') == vocab._padding_token
        assert vocab2.get_token_from_index(1, namespace='b') == vocab._oov_token
        assert vocab2.get_token_from_index(2, namespace='b') == 'b2'
        assert vocab2.get_token_from_index(3, namespace='b') == 'b3'
        assert vocab2.get_token_index(vocab._padding_token, namespace='b') == 0
        assert vocab2.get_token_index(vocab._oov_token, namespace='b') == 1
        assert vocab2.get_token_index('b2', namespace='b') == 2
        assert vocab2.get_token_index('b3', namespace='b') == 3
        assert vocab.get_index_to_token_vocabulary('a') == vocab2.get_index_to_token_vocabulary('a')
        assert vocab.get_index_to_token_vocabulary('b') == vocab2.get_index_to_token_vocabulary('b')

    def test_saving_and_loading_works_with_byte_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = CharacterTokenizer(byte_encoding='utf-8')
        token_indexer = TokenCharactersIndexer(character_tokenizer=tokenizer, min_padding_length=2)
        tokens = [Token(t) for t in ['Øyvind', 'für', '汉字']]
        text_field = TextField(tokens, {'characters': token_indexer})
        dataset = Batch([Instance({'sentence': text_field})])
        vocab = Vocabulary.from_instances(dataset)
        text_field.index(vocab)
        indexed_tokens = deepcopy(text_field._indexed_tokens)
        vocab_dir = self.TEST_DIR / 'vocab_save'
        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)
        text_field2 = TextField(tokens, {'characters': token_indexer})
        text_field2.index(vocab2)
        indexed_tokens2 = deepcopy(text_field2._indexed_tokens)
        assert indexed_tokens == indexed_tokens2

    def test_from_params(self):
        if False:
            i = 10
            return i + 15
        vocab_dir = self.TEST_DIR / 'vocab_save'
        vocab = Vocabulary(non_padded_namespaces=['a', 'c'])
        vocab.add_tokens_to_namespace(['a0', 'a1', 'a2'], namespace='a')
        vocab.add_tokens_to_namespace(['b2', 'b3'], namespace='b')
        vocab.save_to_files(vocab_dir)
        params = Params({'type': 'from_files', 'directory': vocab_dir})
        vocab2 = Vocabulary.from_params(params)
        assert vocab.get_index_to_token_vocabulary('a') == vocab2.get_index_to_token_vocabulary('a')
        assert vocab.get_index_to_token_vocabulary('b') == vocab2.get_index_to_token_vocabulary('b')
        vocab2 = Vocabulary.from_params(Params({}), instances=self.dataset)
        assert vocab2.get_index_to_token_vocabulary('tokens') == {0: '@@PADDING@@', 1: '@@UNKNOWN@@', 2: 'a', 3: 'c', 4: 'b'}
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({}))
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({'type': 'from_files', 'directory': vocab_dir, 'min_count': {'tokens': 2}}))

    def test_from_params_adds_tokens_to_vocab(self):
        if False:
            print('Hello World!')
        vocab = Vocabulary.from_params(Params({'tokens_to_add': {'tokens': ['q', 'x', 'z']}}), instances=self.dataset)
        assert vocab.get_index_to_token_vocabulary('tokens') == {0: '@@PADDING@@', 1: '@@UNKNOWN@@', 2: 'a', 3: 'c', 4: 'b', 5: 'q', 6: 'x', 7: 'z'}

    def test_valid_vocab_extension(self):
        if False:
            print('Hello World!')
        vocab_dir = self.TEST_DIR / 'vocab_save'
        non_padded_namespaces_list = [[], ['tokens']]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_tokens_to_namespace(['d', 'a', 'b'], namespace='tokens')
            text_field = TextField([Token(t) for t in ['a', 'd', 'c', 'e']], {'tokens': SingleIdTokenIndexer('tokens')})
            vocab_dir = self.TEST_DIR / 'vocab_save'
            shutil.rmtree(vocab_dir, ignore_errors=True)
            original_vocab.save_to_files(vocab_dir)
            instances = Batch([Instance({'text': text_field})])
            params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': non_padded_namespaces})
            extended_vocab = Vocabulary.from_params(params, instances=instances)
            extra_count = 2 if extended_vocab.is_padded('tokens') else 0
            assert extended_vocab.get_token_index('d', 'tokens') == 0 + extra_count
            assert extended_vocab.get_token_index('a', 'tokens') == 1 + extra_count
            assert extended_vocab.get_token_index('b', 'tokens') == 2 + extra_count
            assert extended_vocab.get_token_index('c', 'tokens')
            assert extended_vocab.get_token_index('e', 'tokens')
            assert extended_vocab.get_vocab_size('tokens') == 5 + extra_count
        non_padded_namespaces_list = [[], ['tokens1'], ['tokens1', 'tokens2']]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_token_to_namespace('a', namespace='tokens1')
            text_field = TextField([Token(t) for t in ['b']], {'tokens2': SingleIdTokenIndexer('tokens2')})
            instances = Batch([Instance({'text': text_field})])
            vocab_dir = self.TEST_DIR / 'vocab_save'
            shutil.rmtree(vocab_dir, ignore_errors=True)
            original_vocab.save_to_files(vocab_dir)
            params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': non_padded_namespaces})
            extended_vocab = Vocabulary.from_params(params, instances=instances)
            assert len(extended_vocab._token_to_index) == 2
            extra_count = 2 if extended_vocab.is_padded('tokens1') else 0
            assert extended_vocab.get_vocab_size('tokens1') == 1 + extra_count
            extra_count = 2 if extended_vocab.is_padded('tokens2') else 0
            assert extended_vocab.get_vocab_size('tokens2') == 1 + extra_count

    def test_invalid_vocab_extension(self):
        if False:
            for i in range(10):
                print('nop')
        vocab_dir = self.TEST_DIR / 'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=['tokens1'])
        original_vocab.add_tokens_to_namespace(['a', 'b'], namespace='tokens1')
        original_vocab.add_token_to_namespace('p', namespace='tokens2')
        original_vocab.save_to_files(vocab_dir)
        text_field1 = TextField([Token(t) for t in ['a', 'c']], {'tokens1': SingleIdTokenIndexer('tokens1')})
        text_field2 = TextField([Token(t) for t in ['p', 'q', 'r']], {'tokens2': SingleIdTokenIndexer('tokens2')})
        instances = Batch([Instance({'text1': text_field1, 'text2': text_field2})])
        params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': [], 'tokens_to_add': {'tokens1': ['a'], 'tokens2': ['p']}})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)
        params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': ['tokens1'], 'tokens_to_add': {'tokens1': ['a'], 'tokens2': ['p']}})
        Vocabulary.from_params(params, instances=instances)
        params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': ['tokens1', 'tokens2'], 'tokens_to_add': {'tokens1': ['a'], 'tokens2': ['p']}})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)

    def test_from_params_extend_config(self):
        if False:
            i = 10
            return i + 15
        vocab_dir = self.TEST_DIR / 'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=['tokens'])
        original_vocab.add_token_to_namespace('a', namespace='tokens')
        original_vocab.save_to_files(vocab_dir)
        text_field = TextField([Token(t) for t in ['a', 'b']], {'tokens': SingleIdTokenIndexer('tokens')})
        instances = Batch([Instance({'text': text_field})])
        params = Params({'type': 'extend', 'directory': vocab_dir})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params)
        params = Params({'type': 'extend'})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances=instances)

    def test_from_params_valid_vocab_extension_thoroughly(self):
        if False:
            while True:
                i = 10
        '\n        Tests for Valid Vocab Extension thoroughly: Vocab extension is valid\n        when overlapping namespaces have same padding behaviour (padded/non-padded)\n        Summary of namespace paddings in this test:\n        original_vocab namespaces\n            tokens0     padded\n            tokens1     non-padded\n            tokens2     padded\n            tokens3     non-padded\n        instances namespaces\n            tokens0     padded\n            tokens1     non-padded\n            tokens4     padded\n            tokens5     non-padded\n        TypicalExtention example: (of tokens1 namespace)\n        -> original_vocab index2token\n           apple          #0->apple\n           bat            #1->bat\n           cat            #2->cat\n        -> Token to be extended with: cat, an, apple, banana, atom, bat\n        -> extended_vocab: index2token\n           apple           #0->apple\n           bat             #1->bat\n           cat             #2->cat\n           an              #3->an\n           atom            #4->atom\n           banana          #5->banana\n        '
        vocab_dir = self.TEST_DIR / 'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=['tokens1', 'tokens3'])
        original_vocab.add_token_to_namespace('apple', namespace='tokens0')
        original_vocab.add_token_to_namespace('bat', namespace='tokens0')
        original_vocab.add_token_to_namespace('cat', namespace='tokens0')
        original_vocab.add_token_to_namespace('apple', namespace='tokens1')
        original_vocab.add_token_to_namespace('bat', namespace='tokens1')
        original_vocab.add_token_to_namespace('cat', namespace='tokens1')
        original_vocab.add_token_to_namespace('a', namespace='tokens2')
        original_vocab.add_token_to_namespace('b', namespace='tokens2')
        original_vocab.add_token_to_namespace('c', namespace='tokens2')
        original_vocab.add_token_to_namespace('p', namespace='tokens3')
        original_vocab.add_token_to_namespace('q', namespace='tokens3')
        original_vocab.save_to_files(vocab_dir)
        text_field0 = TextField([Token(t) for t in ['cat', 'an', 'apple', 'banana', 'atom', 'bat']], {'tokens0': SingleIdTokenIndexer('tokens0')})
        text_field1 = TextField([Token(t) for t in ['cat', 'an', 'apple', 'banana', 'atom', 'bat']], {'tokens1': SingleIdTokenIndexer('tokens1')})
        text_field4 = TextField([Token(t) for t in ['l', 'm', 'n', 'o']], {'tokens4': SingleIdTokenIndexer('tokens4')})
        text_field5 = TextField([Token(t) for t in ['x', 'y', 'z']], {'tokens5': SingleIdTokenIndexer('tokens5')})
        instances = Batch([Instance({'text0': text_field0, 'text1': text_field1, 'text4': text_field4, 'text5': text_field5})])
        params = Params({'type': 'extend', 'directory': vocab_dir, 'non_padded_namespaces': ['tokens1', 'tokens5']})
        extended_vocab = Vocabulary.from_params(params, instances=instances)
        extended_namespaces = {*extended_vocab._token_to_index}
        assert extended_namespaces == {'tokens{}'.format(i) for i in range(6)}
        assert extended_vocab._non_padded_namespaces == {'tokens1', 'tokens3', 'tokens5'}
        assert extended_vocab.get_vocab_size('tokens1') == 6
        assert extended_vocab.get_vocab_size('tokens0') == 8
        assert extended_vocab.get_vocab_size('tokens2') == original_vocab.get_vocab_size('tokens2')
        assert extended_vocab.get_vocab_size('tokens3') == original_vocab.get_vocab_size('tokens3')
        assert extended_vocab.get_vocab_size('tokens4') == 6
        assert extended_vocab.get_vocab_size('tokens5') == 3
        for (namespace, token2index) in original_vocab._token_to_index.items():
            for (token, _) in token2index.items():
                vocab_index = original_vocab.get_token_index(token, namespace)
                extended_vocab_index = extended_vocab.get_token_index(token, namespace)
                assert vocab_index == extended_vocab_index
        for (namespace, index2token) in original_vocab._index_to_token.items():
            for (index, _) in index2token.items():
                vocab_token = original_vocab.get_token_from_index(index, namespace)
                extended_vocab_token = extended_vocab.get_token_from_index(index, namespace)
                assert vocab_token == extended_vocab_token

    def test_vocab_can_print(self):
        if False:
            print('Hello World!')
        vocab = Vocabulary(non_padded_namespaces=['a', 'c'])
        vocab.add_tokens_to_namespace(['a0', 'a1', 'a2'], namespace='a')
        vocab.add_tokens_to_namespace(['b2', 'b3'], namespace='b')
        print(vocab)

    def test_read_pretrained_words(self):
        if False:
            i = 10
            return i + 15
        words = set('If you think you are too small to make a difference try to sleeping with a mosquito àèìòù'.split(' '))
        base_path = str(self.FIXTURES_ROOT / 'embeddings/fake_embeddings.5d.txt')
        for ext in ['', '.gz', '.xz', '.bz2', '.zip', '.tar.gz']:
            file_path = base_path + ext
            words_read = set(_read_pretrained_tokens(file_path))
            assert words_read == words, f'Wrong words for file {file_path}\n   Read: {sorted(words_read)}\nCorrect: {sorted(words)}'
        base_path = str(self.FIXTURES_ROOT / 'embeddings/multi-file-archive')
        file_path = 'folder/fake_embeddings.5d.txt'
        for ext in ['.zip', '.tar.gz']:
            archive_path = base_path + ext
            embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
            words_read = set(_read_pretrained_tokens(embeddings_file_uri))
            assert words_read == words, f'Wrong words for file {archive_path}\n   Read: {sorted(words_read)}\nCorrect: {sorted(words)}'

    def test_from_instances_exclusive_embeddings_file_inside_archive(self):
        if False:
            return 10
        'Just for ensuring there are no problems when reading pretrained tokens from an archive'
        archive_path = str(self.TEST_DIR / 'embeddings-archive.zip')
        with zipfile.ZipFile(archive_path, 'w') as archive:
            file_path = 'embedding.3d.vec'
            with archive.open(file_path, 'w') as embeddings_file:
                embeddings_file.write('a 1.0 2.3 -1.0\n'.encode('utf-8'))
                embeddings_file.write('b 0.1 0.4 -4.0\n'.encode('utf-8'))
            with archive.open('dummy.vec', 'w') as dummy_file:
                dummy_file.write('c 1.0 2.3 -1.0 3.0\n'.encode('utf-8'))
        embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
        vocab = Vocabulary.from_instances(self.dataset, min_count={'tokens': 4}, pretrained_files={'tokens': embeddings_file_uri}, only_include_pretrained_words=True)
        words = set(vocab.get_index_to_token_vocabulary().values())
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words
        vocab = Vocabulary.from_instances(self.dataset, pretrained_files={'tokens': embeddings_file_uri}, only_include_pretrained_words=True)
        words = set(vocab.get_index_to_token_vocabulary().values())
        assert 'a' in words
        assert 'b' in words
        assert 'c' not in words

    def test_registrability(self):
        if False:
            print('Hello World!')

        @Vocabulary.register('my-vocabulary', constructor='constructor')
        class MyVocabulary(Vocabulary):

            @classmethod
            def constructor(cls):
                if False:
                    i = 10
                    return i + 15
                return MyVocabulary()
        params = Params({'type': 'my-vocabulary'})
        instance = Instance(fields={})
        vocab = Vocabulary.from_params(params=params, instances=[instance])
        assert isinstance(vocab, MyVocabulary)

    def test_max_vocab_size_dict(self):
        if False:
            i = 10
            return i + 15
        params = Params({'max_vocab_size': {'tokens': 1, 'characters': 20}})
        vocab = Vocabulary.from_params(params=params, instances=self.dataset)
        words = vocab.get_index_to_token_vocabulary().values()
        assert len(words) == 3

    def test_max_vocab_size_partial_dict(self):
        if False:
            for i in range(10):
                print('nop')
        indexers = {'tokens': SingleIdTokenIndexer(), 'token_characters': TokenCharactersIndexer(min_padding_length=3)}
        instance = Instance({'text': TextField([Token(w) for w in 'Abc def ghi jkl mno pqr stu vwx yz'.split(' ')], indexers)})
        dataset = Batch([instance])
        params = Params({'max_vocab_size': {'tokens': 1}})
        vocab = Vocabulary.from_params(params=params, instances=dataset)
        assert len(vocab.get_index_to_token_vocabulary('tokens').values()) == 3
        assert len(vocab.get_index_to_token_vocabulary('token_characters').values()) == 28

    def test_min_pretrained_embeddings(self):
        if False:
            return 10
        params = Params({'pretrained_files': {'tokens': str(self.FIXTURES_ROOT / 'embeddings/glove.6B.100d.sample.txt.gz')}, 'min_pretrained_embeddings': {'tokens': 50}})
        vocab = Vocabulary.from_params(params=params, instances=self.dataset)
        assert vocab.get_vocab_size() >= 50
        assert vocab.get_token_index('his') > 1

    def test_custom_padding_oov_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        vocab = Vocabulary(oov_token='[UNK]')
        assert vocab._oov_token == '[UNK]'
        assert vocab._padding_token == '@@PADDING@@'
        vocab = Vocabulary(padding_token='[PAD]')
        assert vocab._oov_token == '@@UNKNOWN@@'
        assert vocab._padding_token == '[PAD]'
        vocab_dir = self.TEST_DIR / 'vocab_save'
        vocab = Vocabulary(oov_token='<UNK>')
        vocab.add_tokens_to_namespace(['a0', 'a1', 'a2'], namespace='a')
        vocab.save_to_files(vocab_dir)
        params = Params({'type': 'from_files', 'directory': vocab_dir, 'oov_token': '<UNK>'})
        vocab = Vocabulary.from_params(params)
        with pytest.raises(AssertionError) as excinfo:
            vocab = Vocabulary.from_params(Params({'type': 'from_files', 'directory': vocab_dir}))
        assert 'OOV token not found!' in str(excinfo.value)

    def test_extend_from_vocab(self):
        if False:
            i = 10
            return i + 15
        vocab1 = Vocabulary(non_padded_namespaces={'1', '2'})
        vocab2 = Vocabulary(non_padded_namespaces={'3'})
        vocab1.add_tokens_to_namespace(['a', 'b', 'c'], namespace='1')
        vocab1.add_tokens_to_namespace(['d', 'e', 'f'], namespace='2')
        vocab2.add_tokens_to_namespace(['c', 'd', 'e'], namespace='1')
        vocab2.add_tokens_to_namespace(['g', 'h', 'i'], namespace='3')
        vocab1.extend_from_vocab(vocab2)
        assert vocab1.get_namespaces() == {'1', '2', '3'}
        assert vocab1._non_padded_namespaces == {'1', '2', '3'}
        assert vocab1.get_token_to_index_vocabulary('1') == {'a': 0, 'b': 1, 'c': 2, '@@PADDING@@': 3, '@@UNKNOWN@@': 4, 'd': 5, 'e': 6}
        assert vocab1.get_token_to_index_vocabulary('2') == {'d': 0, 'e': 1, 'f': 2}
        assert vocab1.get_token_to_index_vocabulary('3') == {'g': 0, 'h': 1, 'i': 2}

    def test_extend_helper(self):
        if False:
            for i in range(10):
                print('nop')
        vocab = Vocabulary()
        counter = {'a': {}, 'b': {'test': 0}, 'c': {'test': 1}}
        min_count = {'c': -1, 'd': 0}
        with pytest.raises(ConfigurationError):
            vocab._extend(counter, min_count)
        with pytest.raises(ConfigurationError):
            vocab._extend(None, min_count)
        counter['d'] = {}
        try:
            vocab._extend(counter, min_count)
        except ConfigurationError:
            pytest.fail('Unexpected ConfigurationError')

class TestVocabularyFromFilesWithArchive(AllenNlpTestCase):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.tar_archive = self.TEST_DIR / 'vocab.tar.gz'
        self.zip_archive = self.TEST_DIR / 'vocab.zip'
        self.model_archive = self.TEST_DIR / 'model.tar.gz'
        shutil.copyfile(self.FIXTURES_ROOT / 'data' / 'vocab.tar.gz', self.tar_archive)
        shutil.copyfile(self.FIXTURES_ROOT / 'data' / 'vocab.zip', self.zip_archive)
        shutil.copyfile(self.FIXTURES_ROOT / 'simple_tagger' / 'serialization' / 'model.tar.gz', self.model_archive)

    def test_from_files_with_zip_archive(self):
        if False:
            i = 10
            return i + 15
        vocab = Vocabulary.from_files(str(self.zip_archive))
        vocab.get_namespaces() == {'tokens'}
        assert vocab.get_token_from_index(3, namespace='tokens') == ','

    def test_from_files_with_tar_archive(self):
        if False:
            return 10
        vocab = Vocabulary.from_files(str(self.tar_archive))
        vocab.get_namespaces() == {'tokens'}
        assert vocab.get_token_from_index(3, namespace='tokens') == ','

    def test_from_files_with_model_archive(self):
        if False:
            return 10
        vocab = Vocabulary.from_files(str(self.model_archive))
        vocab.get_namespaces() == {'tokens', 'labels'}
        assert vocab.get_token_from_index(3, namespace='tokens') == 'u.n.'

class TestVocabularyFromPretrainedTransformer(AllenNlpTestCase):

    @pytest.mark.parametrize('model_name', ['bert-base-cased', 'roberta-base'])
    def test_from_pretrained_transformer(self, model_name):
        if False:
            i = 10
            return i + 15
        namespace = 'tokens'
        from allennlp.common import cached_transformers
        tokenizer = cached_transformers.get_tokenizer(model_name)
        vocab = Vocabulary.from_pretrained_transformer(model_name, namespace=namespace)
        assert vocab._token_to_index[namespace] == tokenizer.get_vocab()
        vocab.save_to_files(self.TEST_DIR / 'vocab')
        vocab1 = Vocabulary.from_files(self.TEST_DIR / 'vocab')
        assert vocab1._token_to_index[namespace] == tokenizer.get_vocab()

class TestVocabularyFromPretrainedTransformerAndInstances(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        token_indexer_1 = SingleIdTokenIndexer('namespace_1')
        text_field_1 = TextField([Token(t) for t in ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c']], {'namespace_1': token_indexer_1})
        single_field_instance = Instance({'text': text_field_1})
        self.single_namespace_dataset = Batch([single_field_instance])
        token_indexer_2 = SingleIdTokenIndexer('namespace_2')
        text_field_2 = TextField([Token(t) for t in ['d', 'd', 'd', 'd', 'e', 'e', 'f', 'f', 'f']], {'namespace_2': token_indexer_2})
        multiple_field_instance = Instance({'first_text': text_field_1, 'second_text': text_field_2})
        self.multiple_namespace_dataset = Batch([multiple_field_instance])

    @staticmethod
    def _get_expected_vocab(dataset, namespace, model_name):
        if False:
            while True:
                i = 10
        vocab_from_instances = Vocabulary.from_instances(dataset)
        instance_tokens = set(vocab_from_instances._token_to_index[namespace].keys())
        transformer_tokens = set(Vocabulary.from_pretrained_transformer(model_name, namespace)._token_to_index[namespace].keys())
        return instance_tokens.union(transformer_tokens)

    def _get_expected_vocab_size(self, dataset, namespace, model_name):
        if False:
            return 10
        return len(self._get_expected_vocab(dataset, namespace, model_name))

    @pytest.mark.parametrize('model_name', ['bert-base-cased', 'roberta-base'])
    def test_with_single_namespace_and_single_model(self, model_name):
        if False:
            while True:
                i = 10
        dataset = self.single_namespace_dataset
        namespace = 'namespace_1'
        expected_vocab_size = self._get_expected_vocab_size(dataset, namespace, model_name)
        vocab = Vocabulary.from_pretrained_transformer_and_instances(dataset, {namespace: model_name})
        assert vocab.get_vocab_size(namespace) == expected_vocab_size

    @pytest.mark.parametrize('model_name', ['bert-base-cased', 'roberta-base'])
    def test_only_updates_single_namespace_when_multiple_present(self, model_name):
        if False:
            return 10
        dataset = self.multiple_namespace_dataset
        namespace1 = 'namespace_1'
        namespace2 = 'namespace_2'
        namespace1_vocab_size = self._get_expected_vocab_size(dataset, namespace1, model_name)
        namespace2_vocab_size = Vocabulary.from_instances(dataset).get_vocab_size('namespace_2')
        vocab = Vocabulary.from_pretrained_transformer_and_instances(dataset, {namespace1: model_name})
        assert vocab.get_vocab_size(namespace1) == namespace1_vocab_size
        assert vocab.get_vocab_size(namespace2) == namespace2_vocab_size

    @pytest.mark.parametrize('namespace1_model_name', ['bert-base-cased', 'roberta-base'])
    @pytest.mark.parametrize('namespace2_model_name', ['bert-base-cased', 'roberta-base'])
    def test_with_different_models_per_namespace(self, namespace1_model_name, namespace2_model_name):
        if False:
            print('Hello World!')
        dataset = self.multiple_namespace_dataset
        namespace1 = 'namespace_1'
        namespace2 = 'namespace_2'
        namespace1_vocab_size = self._get_expected_vocab_size(dataset, namespace1, namespace1_model_name)
        namespace2_vocab_size = self._get_expected_vocab_size(dataset, namespace2, namespace2_model_name)
        vocab = Vocabulary.from_pretrained_transformer_and_instances(dataset, {namespace1: namespace1_model_name, namespace2: namespace2_model_name})
        assert vocab.get_vocab_size(namespace1) == namespace1_vocab_size
        assert vocab.get_vocab_size(namespace2) == namespace2_vocab_size