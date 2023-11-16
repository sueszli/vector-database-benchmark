import os
import unittest
from transformers import LxmertTokenizer, LxmertTokenizerFast
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class LxmertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LxmertTokenizer
    rust_tokenizer_class = LxmertTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',', 'low', 'lowest']
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        if False:
            while True:
                i = 10
        input_text = 'UNwantéd,running'
        output_text = 'unwanted, running'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            while True:
                i = 10
        tokenizer = self.tokenizer_class(self.vocab_file)
        tokens = tokenizer.tokenize('UNwantéd,running')
        self.assertListEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_rust_and_python_full_tokenizers(self):
        if False:
            print('Hello World!')
        if not self.test_rust_tokenizer:
            return
        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()
        sequence = 'I was born in 92000, and this is falsé.'
        tokens = tokenizer.tokenize(sequence)
        rust_tokens = rust_tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, rust_tokens)
        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)
        rust_tokenizer = self.get_rust_tokenizer()
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)