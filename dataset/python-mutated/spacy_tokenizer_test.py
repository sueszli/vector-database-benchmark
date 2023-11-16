import spacy
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import Token, SpacyTokenizer

class TestSpacyTokenizer(AllenNlpTestCase):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.word_tokenizer = SpacyTokenizer()

    def test_tokenize_handles_complex_punctuation(self):
        if False:
            return 10
        sentence = 'this (sentence) has \'crazy\' "punctuation".'
        expected_tokens = ['this', '(', 'sentence', ')', 'has', "'", 'crazy', "'", '"', 'punctuation', '"', '.']
        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens
        for token in tokens:
            start = token.idx
            end = start + len(token.text)
            assert sentence[start:end] == token.text

    def test_tokenize_handles_contraction(self):
        if False:
            while True:
                i = 10
        sentence = "it ain't joe's problem; would been yesterday"
        expected_tokens = ['it', 'ai', "n't", 'joe', "'s", 'problem', ';', 'would', 'been', 'yesterday']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_multiple_contraction(self):
        if False:
            i = 10
            return i + 15
        sentence = "wouldn't've"
        expected_tokens = ['would', "n't", "'ve"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        if False:
            return 10
        sentence = "the jones' house"
        expected_tokens = ['the', 'jones', "'", 'house']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_removes_whitespace_tokens(self):
        if False:
            while True:
                i = 10
        sentence = "the\n jones'   house  \x0b  55"
        expected_tokens = ['the', 'jones', "'", 'house', '55']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        if False:
            while True:
                i = 10
        sentence = 'Mr. and Mrs. Jones, etc., went to, e.g., the store'
        expected_tokens = ['Mr.', 'and', 'Mrs.', 'Jones', ',', 'etc', '.', ',', 'went', 'to', ',', 'e.g.', ',', 'the', 'store']
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        if False:
            i = 10
            return i + 15
        sentences = ['This is     a sentence', "This isn't a sentence.", "This is the 3rd     sentence.Here's the 'fourth' sentence."]
        batch_split = self.word_tokenizer.batch_tokenize(sentences)
        separately_split = [self.word_tokenizer.tokenize(sentence) for sentence in sentences]
        assert len(batch_split) == len(separately_split)
        for (batch_sentence, separate_sentence) in zip(batch_split, separately_split):
            assert len(batch_sentence) == len(separate_sentence)
            for (batch_word, separate_word) in zip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_keep_spacy_tokens(self):
        if False:
            i = 10
            return i + 15
        word_tokenizer = SpacyTokenizer()
        sentence = 'This should be an allennlp Token'
        tokens = word_tokenizer.tokenize(sentence)
        assert tokens
        assert all((isinstance(token, Token) for token in tokens))
        word_tokenizer = SpacyTokenizer(keep_spacy_tokens=True)
        sentence = 'This should be a spacy Token'
        tokens = word_tokenizer.tokenize(sentence)
        assert tokens
        assert all((isinstance(token, spacy.tokens.Token) for token in tokens))

    def test_to_params(self):
        if False:
            return 10
        tokenizer = SpacyTokenizer()
        params = tokenizer.to_params()
        assert isinstance(params, Params)
        assert params.params == {'type': 'spacy', 'language': tokenizer._language, 'pos_tags': tokenizer._pos_tags, 'parse': tokenizer._parse, 'ner': tokenizer._ner, 'keep_spacy_tokens': tokenizer._keep_spacy_tokens, 'split_on_spaces': tokenizer._split_on_spaces, 'start_tokens': tokenizer._start_tokens, 'end_tokens': tokenizer._end_tokens}