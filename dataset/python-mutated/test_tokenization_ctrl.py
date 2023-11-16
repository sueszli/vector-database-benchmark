import json
import os
import unittest
from transformers.models.ctrl.tokenization_ctrl import VOCAB_FILES_NAMES, CTRLTokenizer
from ...test_tokenization_common import TokenizerTesterMixin

class CTRLTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = CTRLTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        vocab = ['adapt', 're@@', 'a@@', 'apt', 'c@@', 't', '<unk>']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['#version: 0.2', 'a p', 'ap t</w>', 'r e', 'a d', 'ad apt</w>', '']
        self.special_tokens_map = {'unk_token': '<unk>'}
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['merges_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(vocab_tokens) + '\n')
        with open(self.merges_file, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(merges))

    def get_tokenizer(self, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(self.special_tokens_map)
        return CTRLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            while True:
                i = 10
        input_text = 'adapt react readapt apt'
        output_text = 'adapt react readapt apt'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = CTRLTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'adapt react readapt apt'
        bpe_tokens = 'adapt re@@ a@@ c@@ t re@@ adapt apt'.split()
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)