from collections import defaultdict
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

class CharacterTokenIndexerTest(AllenNlpTestCase):

    def test_count_vocab_items_respects_casing(self):
        if False:
            print('Hello World!')
        indexer = TokenCharactersIndexer('characters', min_padding_length=5)
        counter = defaultdict(lambda : defaultdict(int))
        indexer.count_vocab_items(Token('Hello'), counter)
        indexer.count_vocab_items(Token('hello'), counter)
        assert counter['characters'] == {'h': 1, 'H': 1, 'e': 2, 'l': 4, 'o': 2}
        indexer = TokenCharactersIndexer('characters', CharacterTokenizer(lowercase_characters=True), min_padding_length=5)
        counter = defaultdict(lambda : defaultdict(int))
        indexer.count_vocab_items(Token('Hello'), counter)
        indexer.count_vocab_items(Token('hello'), counter)
        assert counter['characters'] == {'h': 2, 'e': 2, 'l': 4, 'o': 2}

    def test_as_array_produces_token_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        indexer = TokenCharactersIndexer('characters', min_padding_length=1)
        padded_tokens = indexer.as_padded_tensor_dict({'token_characters': [[1, 2, 3, 4, 5], [1, 2, 3], [1]]}, padding_lengths={'token_characters': 4, 'num_token_characters': 10})
        assert padded_tokens['token_characters'].tolist() == [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_tokens_to_indices_produces_correct_characters(self):
        if False:
            while True:
                i = 10
        vocab = Vocabulary()
        vocab.add_token_to_namespace('A', namespace='characters')
        vocab.add_token_to_namespace('s', namespace='characters')
        vocab.add_token_to_namespace('e', namespace='characters')
        vocab.add_token_to_namespace('n', namespace='characters')
        vocab.add_token_to_namespace('t', namespace='characters')
        vocab.add_token_to_namespace('c', namespace='characters')
        indexer = TokenCharactersIndexer('characters', min_padding_length=1)
        indices = indexer.tokens_to_indices([Token('sentential')], vocab)
        assert indices == {'token_characters': [[3, 4, 5, 6, 4, 5, 6, 1, 1, 1]]}

    def test_start_and_end_tokens(self):
        if False:
            i = 10
            return i + 15
        vocab = Vocabulary()
        vocab.add_token_to_namespace('A', namespace='characters')
        vocab.add_token_to_namespace('s', namespace='characters')
        vocab.add_token_to_namespace('e', namespace='characters')
        vocab.add_token_to_namespace('n', namespace='characters')
        vocab.add_token_to_namespace('t', namespace='characters')
        vocab.add_token_to_namespace('c', namespace='characters')
        vocab.add_token_to_namespace('<', namespace='characters')
        vocab.add_token_to_namespace('>', namespace='characters')
        vocab.add_token_to_namespace('/', namespace='characters')
        indexer = TokenCharactersIndexer('characters', start_tokens=['<s>'], end_tokens=['</s>'], min_padding_length=1)
        indices = indexer.tokens_to_indices([Token('sentential')], vocab)
        assert indices == {'token_characters': [[8, 3, 9], [3, 4, 5, 6, 4, 5, 6, 1, 1, 1], [8, 10, 3, 9]]}

    def test_min_padding_length(self):
        if False:
            i = 10
            return i + 15
        sentence = 'AllenNLP is awesome .'
        tokens = [Token(token) for token in sentence.split(' ')]
        vocab = Vocabulary()
        vocab.add_token_to_namespace('A', namespace='characters')
        vocab.add_token_to_namespace('l', namespace='characters')
        vocab.add_token_to_namespace('e', namespace='characters')
        vocab.add_token_to_namespace('n', namespace='characters')
        vocab.add_token_to_namespace('N', namespace='characters')
        vocab.add_token_to_namespace('L', namespace='characters')
        vocab.add_token_to_namespace('P', namespace='characters')
        vocab.add_token_to_namespace('i', namespace='characters')
        vocab.add_token_to_namespace('s', namespace='characters')
        vocab.add_token_to_namespace('a', namespace='characters')
        vocab.add_token_to_namespace('w', namespace='characters')
        vocab.add_token_to_namespace('o', namespace='characters')
        vocab.add_token_to_namespace('m', namespace='characters')
        vocab.add_token_to_namespace('.', namespace='characters')
        indexer = TokenCharactersIndexer('characters', min_padding_length=10)
        indices = indexer.tokens_to_indices(tokens, vocab)
        padded = indexer.as_padded_tensor_dict(indices, indexer.get_padding_lengths(indices))
        assert padded['token_characters'].tolist() == [[2, 3, 3, 4, 5, 6, 7, 8, 0, 0], [9, 10, 0, 0, 0, 0, 0, 0, 0, 0], [11, 12, 4, 10, 13, 14, 4, 0, 0, 0], [15, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_warn_min_padding_length(self):
        if False:
            i = 10
            return i + 15
        with pytest.warns(UserWarning, match='using the default value \\(0\\) of `min_padding_length`'):
            TokenCharactersIndexer('characters')