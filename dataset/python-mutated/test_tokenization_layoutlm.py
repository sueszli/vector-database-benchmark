import os
import unittest
from transformers import LayoutLMTokenizer, LayoutLMTokenizerFast
from transformers.models.layoutlm.tokenization_layoutlm import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class LayoutLMTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LayoutLMTokenizer
    rust_tokenizer_class = LayoutLMTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',', 'low', 'lowest']
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        if False:
            return 10
        return LayoutLMTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        if False:
            i = 10
            return i + 15
        input_text = 'UNwantéd,running'
        output_text = 'unwanted, running'
        return (input_text, output_text)

    def test_full_tokenizer(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = self.tokenizer_class(self.vocab_file)
        tokens = tokenizer.tokenize('UNwantéd,running')
        self.assertListEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_special_tokens_as_you_expect(self):
        if False:
            while True:
                i = 10
        'If you are training a seq2seq model that expects a decoder_prefix token make sure it is prepended to decoder_input_ids'
        pass