import os
import unittest
from transformers import MPNetTokenizerFast
from transformers.models.mpnet.tokenization_mpnet import VOCAB_FILES_NAMES, MPNetTokenizer
from transformers.testing_utils import require_tokenizers, slow
from ...test_tokenization_common import TokenizerTesterMixin

@require_tokenizers
class MPNetTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MPNetTokenizer
    rust_tokenizer_class = MPNetTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',', 'low', 'lowest']
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES['vocab_file'])
        with open(self.vocab_file, 'w', encoding='utf-8') as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        if False:
            print('Hello World!')
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
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

    @slow
    def test_sequence_builders(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = self.tokenizer_class.from_pretrained('microsoft/mpnet-base')
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == [0] + text + [2]
        assert encoded_pair == [0] + text + [2] + [2] + text_2 + [2]