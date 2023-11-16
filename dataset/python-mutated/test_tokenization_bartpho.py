import os
import unittest
from transformers.models.bartpho.tokenization_bartpho import VOCAB_FILES_NAMES, BartphoTokenizer
from transformers.testing_utils import get_tests_dir
from ...test_tokenization_common import TokenizerTesterMixin
SAMPLE_VOCAB = get_tests_dir('fixtures/test_sentencepiece_bpe.model')

class BartphoTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BartphoTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        vocab = ['▁This', '▁is', '▁a', '▁t', 'est']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        self.special_tokens_map = {'unk_token': '<unk>'}
        self.monolingual_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['monolingual_vocab_file'])
        with open(self.monolingual_vocab_file, 'w', encoding='utf-8') as fp:
            for token in vocab_tokens:
                fp.write(f'{token} {vocab_tokens[token]}\n')
        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, self.monolingual_vocab_file, **self.special_tokens_map)
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.update(self.special_tokens_map)
        return BartphoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            while True:
                i = 10
        input_text = 'This is a là test'
        output_text = 'This is a<unk><unk> test'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            i = 10
            return i + 15
        tokenizer = BartphoTokenizer(SAMPLE_VOCAB, self.monolingual_vocab_file, **self.special_tokens_map)
        text = 'This is a là test'
        bpe_tokens = '▁This ▁is ▁a ▁l à ▁t est'.split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [4, 5, 6, 3, 3, 7, 8, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)