import os
import unittest
from transformers import FunnelTokenizer, FunnelTokenizerFast
from transformers.models.funnel.tokenization_funnel import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class FunnelTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FunnelTokenizer
    rust_tokenizer_class = FunnelTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        vocab_tokens = ['<unk>', '<cls>', '<sep>', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',', 'low', 'lowest']
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    def get_tokenizer(self, **kwargs):
        if False:
            return 10
        return FunnelTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        if False:
            return 10
        return FunnelTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

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

    def test_token_type_ids(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            inputs = tokenizer('UNwantéd,running')
            sentence_len = len(inputs['input_ids']) - 1
            self.assertListEqual(inputs['token_type_ids'], [2] + [0] * sentence_len)
            inputs = tokenizer('UNwantéd,running', 'UNwantéd,running')
            self.assertListEqual(inputs['token_type_ids'], [2] + [0] * sentence_len + [1] * sentence_len)