import os
import unittest
from transformers import BatchEncoding
from transformers.models.bert.tokenization_bert import BasicTokenizer, WordpieceTokenizer, _is_control, _is_punctuation, _is_whitespace
from transformers.models.prophetnet.tokenization_prophetnet import VOCAB_FILES_NAMES, ProphetNetTokenizer
from transformers.testing_utils import require_torch, slow
from ...test_tokenization_common import TokenizerTesterMixin

class ProphetNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = ProphetNetTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',', 'low', 'lowest']
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
            i = 10
            return i + 15
        tokenizer = self.tokenizer_class(self.vocab_file)
        tokens = tokenizer.tokenize('UNwantéd,running')
        self.assertListEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing'])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

    def test_chinese(self):
        if False:
            print('Hello World!')
        tokenizer = BasicTokenizer()
        self.assertListEqual(tokenizer.tokenize('ah博推zz'), ['ah', '博', '推', 'zz'])

    def test_basic_tokenizer_lower(self):
        if False:
            while True:
                i = 10
        tokenizer = BasicTokenizer(do_lower_case=True)
        self.assertListEqual(tokenizer.tokenize(' \tHeLLo!how  \n Are yoU?  '), ['hello', '!', 'how', 'are', 'you', '?'])
        self.assertListEqual(tokenizer.tokenize('Héllo'), ['hello'])

    def test_basic_tokenizer_lower_strip_accents_false(self):
        if False:
            while True:
                i = 10
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=False)
        self.assertListEqual(tokenizer.tokenize(' \tHäLLo!how  \n Are yoU?  '), ['hällo', '!', 'how', 'are', 'you', '?'])
        self.assertListEqual(tokenizer.tokenize('Héllo'), ['héllo'])

    def test_basic_tokenizer_lower_strip_accents_true(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = BasicTokenizer(do_lower_case=True, strip_accents=True)
        self.assertListEqual(tokenizer.tokenize(' \tHäLLo!how  \n Are yoU?  '), ['hallo', '!', 'how', 'are', 'you', '?'])
        self.assertListEqual(tokenizer.tokenize('Héllo'), ['hello'])

    def test_basic_tokenizer_lower_strip_accents_default(self):
        if False:
            i = 10
            return i + 15
        tokenizer = BasicTokenizer(do_lower_case=True)
        self.assertListEqual(tokenizer.tokenize(' \tHäLLo!how  \n Are yoU?  '), ['hallo', '!', 'how', 'are', 'you', '?'])
        self.assertListEqual(tokenizer.tokenize('Héllo'), ['hello'])

    def test_basic_tokenizer_no_lower(self):
        if False:
            for i in range(10):
                print('nop')
        tokenizer = BasicTokenizer(do_lower_case=False)
        self.assertListEqual(tokenizer.tokenize(' \tHeLLo!how  \n Are yoU?  '), ['HeLLo', '!', 'how', 'Are', 'yoU', '?'])

    def test_basic_tokenizer_no_lower_strip_accents_false(self):
        if False:
            return 10
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=False)
        self.assertListEqual(tokenizer.tokenize(' \tHäLLo!how  \n Are yoU?  '), ['HäLLo', '!', 'how', 'Are', 'yoU', '?'])

    def test_basic_tokenizer_no_lower_strip_accents_true(self):
        if False:
            return 10
        tokenizer = BasicTokenizer(do_lower_case=False, strip_accents=True)
        self.assertListEqual(tokenizer.tokenize(' \tHäLLo!how  \n Are yoU?  '), ['HaLLo', '!', 'how', 'Are', 'yoU', '?'])

    def test_basic_tokenizer_respects_never_split_tokens(self):
        if False:
            while True:
                i = 10
        tokenizer = BasicTokenizer(do_lower_case=False, never_split=['[UNK]'])
        self.assertListEqual(tokenizer.tokenize(' \tHeLLo!how  \n Are yoU? [UNK]'), ['HeLLo', '!', 'how', 'Are', 'yoU', '?', '[UNK]'])

    def test_wordpiece_tokenizer(self):
        if False:
            while True:
                i = 10
        vocab_tokens = ['[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing']
        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')
        self.assertListEqual(tokenizer.tokenize(''), [])
        self.assertListEqual(tokenizer.tokenize('unwanted running'), ['un', '##want', '##ed', 'runn', '##ing'])
        self.assertListEqual(tokenizer.tokenize('unwantedX running'), ['[UNK]', 'runn', '##ing'])

    @require_torch
    def test_prepare_batch(self):
        if False:
            return 10
        tokenizer = self.tokenizer_class.from_pretrained('microsoft/prophetnet-large-uncased')
        src_text = ['A long paragraph for summarization.', 'Another paragraph for summarization.']
        expected_src_tokens = [1037, 2146, 20423, 2005, 7680, 7849, 3989, 1012, 102]
        batch = tokenizer(src_text, padding=True, return_tensors='pt')
        self.assertIsInstance(batch, BatchEncoding)
        result = list(batch.input_ids.numpy()[0])
        self.assertListEqual(expected_src_tokens, result)
        self.assertEqual((2, 9), batch.input_ids.shape)
        self.assertEqual((2, 9), batch.attention_mask.shape)

    def test_is_whitespace(self):
        if False:
            print('Hello World!')
        self.assertTrue(_is_whitespace(' '))
        self.assertTrue(_is_whitespace('\t'))
        self.assertTrue(_is_whitespace('\r'))
        self.assertTrue(_is_whitespace('\n'))
        self.assertTrue(_is_whitespace('\xa0'))
        self.assertFalse(_is_whitespace('A'))
        self.assertFalse(_is_whitespace('-'))

    def test_is_control(self):
        if False:
            while True:
                i = 10
        self.assertTrue(_is_control('\x05'))
        self.assertFalse(_is_control('A'))
        self.assertFalse(_is_control(' '))
        self.assertFalse(_is_control('\t'))
        self.assertFalse(_is_control('\r'))

    def test_is_punctuation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(_is_punctuation('-'))
        self.assertTrue(_is_punctuation('$'))
        self.assertTrue(_is_punctuation('`'))
        self.assertTrue(_is_punctuation('.'))
        self.assertFalse(_is_punctuation('A'))
        self.assertFalse(_is_punctuation(' '))

    @slow
    def test_sequence_builders(self):
        if False:
            i = 10
            return i + 15
        tokenizer = self.tokenizer_class.from_pretrained('microsoft/prophetnet-large-uncased')
        text = tokenizer.encode('sequence builders', add_special_tokens=False)
        text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
        assert encoded_sentence == text + [102]
        assert encoded_pair == text + [102] + text_2 + [102]