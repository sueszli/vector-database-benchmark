import os
import unittest
from transformers.models.bertweet.tokenization_bertweet import VOCAB_FILES_NAMES, BertweetTokenizer
from ...test_tokenization_common import TokenizerTesterMixin

class BertweetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BertweetTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        vocab = ['I', 'm', 'V@@', 'R@@', 'r', 'e@@']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['#version: 0.2', 'a m</w>']
        self.special_tokens_map = {'unk_token': '<unk>'}
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as fp:
            for token in vocab_tokens:
                fp.write(f'{token} {vocab_tokens[token]}\n')
        with open(self.merges_file, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(merges))

    def get_tokenizer(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs.update(self.special_tokens_map)
        return BertweetTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            print('Hello World!')
        input_text = 'I am VinAI Research'
        output_text = 'I <unk> m V<unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            return 10
        tokenizer = BertweetTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'I am VinAI Research'
        bpe_tokens = 'I a@@ m V@@ i@@ n@@ A@@ I R@@ e@@ s@@ e@@ a@@ r@@ c@@ h'.split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [4, 3, 5, 6, 3, 3, 3, 4, 7, 9, 3, 9, 3, 3, 3, 3, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)