import os
import unittest
from transformers.models.phobert.tokenization_phobert import VOCAB_FILES_NAMES, PhobertTokenizer
from ...test_tokenization_common import TokenizerTesterMixin

class PhobertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = PhobertTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        if False:
            return 10
        super().setUp()
        vocab = ['T@@', 'i', 'I', 'R@@', 'r', 'e@@']
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ['#version: 0.2', 'l à</w>']
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
            print('Hello World!')
        kwargs.update(self.special_tokens_map)
        return PhobertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        input_text = 'Tôi là VinAI Research'
        output_text = 'T<unk> i <unk> <unk> <unk> <unk> <unk> <unk> I Re<unk> e<unk> <unk> <unk> <unk>'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            i = 10
            return i + 15
        tokenizer = PhobertTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = 'Tôi là VinAI Research'
        bpe_tokens = 'T@@ ô@@ i l@@ à V@@ i@@ n@@ A@@ I R@@ e@@ s@@ e@@ a@@ r@@ c@@ h'.split()
        tokens = tokenizer.tokenize(text)
        print(tokens)
        self.assertListEqual(tokens, bpe_tokens)
        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [4, 3, 5, 3, 3, 3, 3, 3, 3, 6, 7, 9, 3, 9, 3, 3, 3, 3, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)