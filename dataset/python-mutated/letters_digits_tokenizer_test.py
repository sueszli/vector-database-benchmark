from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token, LettersDigitsTokenizer

class TestLettersDigitsTokenizer(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.word_tokenizer = LettersDigitsTokenizer()

    def test_tokenize_handles_complex_punctuation(self):
        if False:
            while True:
                i = 10
        sentence = 'this (sentence) has \'crazy\' "punctuation".'
        expected_tokens = ['this', '(', 'sentence', ')', 'has', "'", 'crazy', "'", '"', 'punctuation', '"', '.']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_unicode_letters(self):
        if False:
            while True:
                i = 10
        sentence = 'HAL9000   and    Ångström'
        expected_tokens = [Token('HAL', 0), Token('9000', 3), Token('and', 10), Token('Ångström', 17)]
        tokens = self.word_tokenizer.tokenize(sentence)
        assert [t.text for t in tokens] == [t.text for t in expected_tokens]
        assert [t.idx for t in tokens] == [t.idx for t in expected_tokens]

    def test_tokenize_handles_splits_all_punctuation(self):
        if False:
            while True:
                i = 10
        sentence = "wouldn't.[have] -3.45(m^2)"
        expected_tokens = ['wouldn', "'", 't', '.', '[', 'have', ']', '-', '3', '.', '45', '(', 'm', '^', '2', ')']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens