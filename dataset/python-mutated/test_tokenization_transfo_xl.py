import os
import unittest
from transformers.models.transfo_xl.tokenization_transfo_xl import VOCAB_FILES_NAMES, TransfoXLTokenizer
from ...test_tokenization_common import TokenizerTesterMixin

class TransfoXLTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = TransfoXLTokenizer
    test_rust_tokenizer = False
    test_seq2seq = False

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        vocab_tokens = ['<unk>', '[CLS]', '[SEP]', 'want', 'unwanted', 'wa', 'un', 'running', ',', 'low', 'l']
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['lower_case'] = True
        return TransfoXLTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            for i in range(10):
                print('nop')
        input_text = '<unk> UNwanted , running'
        output_text = '<unk> unwanted, running'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = TransfoXLTokenizer(vocab_file=self.vocab_file, lower_case=True)
        tokens = tokenizer.tokenize('<unk> UNwanted , running')
        self.assertListEqual(tokens, ['<unk>', 'unwanted', ',', 'running'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [0, 4, 8, 7])

    def test_full_tokenizer_lower(self):
        if False:
            return 10
        tokenizer = TransfoXLTokenizer(lower_case=True)
        self.assertListEqual(tokenizer.tokenize(' \tHeLLo ! how  \n Are yoU ?  '), ['hello', '!', 'how', 'are', 'you', '?'])

    def test_full_tokenizer_no_lower(self):
        if False:
            i = 10
            return i + 15
        tokenizer = TransfoXLTokenizer(lower_case=False)
        self.assertListEqual(tokenizer.tokenize(' \tHeLLo ! how  \n Are yoU ?  '), ['HeLLo', '!', 'how', 'Are', 'yoU', '?'])

    def test_full_tokenizer_moses_numbers(self):
        if False:
            while True:
                i = 10
        tokenizer = TransfoXLTokenizer(lower_case=False)
        text_in = "Hello (bracket) and side-scrolled [and] Henry's $5,000 with 3.34 m. What's up!?"
        tokens_out = ['Hello', '(', 'bracket', ')', 'and', 'side', '@-@', 'scrolled', '[', 'and', ']', 'Henry', "'s", '$', '5', '@,@', '000', 'with', '3', '@.@', '34', 'm', '.', 'What', "'s", 'up', '!', '?']
        self.assertListEqual(tokenizer.tokenize(text_in), tokens_out)
        self.assertEqual(tokenizer.convert_tokens_to_string(tokens_out), text_in)

    def test_move_added_token(self):
        if False:
            i = 10
            return i + 15
        tokenizer = self.get_tokenizer()
        original_len = len(tokenizer)
        tokenizer.add_tokens(['new1', 'new2'])
        tokenizer.move_added_token('new1', 1)
        self.assertEqual(len(tokenizer), original_len + 2)
        self.assertEqual(tokenizer.encode('new1'), [1])
        self.assertEqual(tokenizer.decode([1]), 'new1')